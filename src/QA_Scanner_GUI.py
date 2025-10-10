"""
QA Scanner GUI Methods
These methods can be integrated into TranslatorGUI or used standalone
"""

import os
import sys
import re
import json
from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QLabel, QPushButton, 
                               QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, 
                               QCheckBox, QSpinBox, QSlider, QTextEdit, QScrollArea,
                               QRadioButton, QButtonGroup, QGroupBox, QComboBox,
                               QFileDialog, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QPixmap, QIcon
import threading
import traceback

# WindowManager and UIHelper removed - not needed in PySide6
# Qt handles window management and UI utilities automatically
scan_html_folder = None  # Will be lazy-loaded from translator_gui


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
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox with all checkmarks disabled"""
        from PySide6.QtWidgets import QCheckBox
        
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet("""
            QCheckBox { 
                color: white; 
            }
            QCheckBox::indicator {
                background-image: none;
                image: none;
                content: none;
                text: none;
            }
            QCheckBox::indicator:checked {
                background-image: none;
                image: none;
                content: none;
                text: none;
            }
        """)
        return checkbox
    
    def run_qa_scan(self, mode_override=None, non_interactive=False, preselected_files=None):
        """Run QA scan with mode selection and settings"""
        # Removed loading screen - initialize directly for smoother experience
        try:
            # Start a brief auto-scroll delay so first log lines are readable
            try:
                import time as _time
                if hasattr(self, '_start_autoscroll_delay'):
                    self._start_autoscroll_delay(100)
                elif hasattr(self, '_autoscroll_delay_until'):
                    self._autoscroll_delay_until = _time.time() + 0.6
            except Exception:
                pass
            
            if not self._lazy_load_modules():
                self.append_log("❌ Failed to load QA scanner modules")
                return
            
            # Check for scan_html_folder in the global scope from translator_gui
            import sys
            translator_module = sys.modules.get('translator_gui')
            if translator_module is None or not hasattr(translator_module, 'scan_html_folder') or translator_module.scan_html_folder is None:
                self.append_log("❌ QA scanner module is not available")
                QMessageBox.critical(None, "Module Error", "QA scanner module is not available.")
                return
            
            if hasattr(self, 'qa_thread') and self.qa_thread and self.qa_thread.is_alive():
                self.stop_requested = True
                self.append_log("⛔ QA scan stop requested.")
                return
            
            self.append_log("✅ QA scanner initialized successfully")
            
        except Exception as e:
            self.append_log(f"❌ Error initializing QA scanner: {e}")
            return
        
        # Load QA scanner settings from config
        qa_settings = self.config.get('qa_scanner_settings', {
            'foreign_char_threshold': 10,
            'excluded_characters': '',
            'target_language': 'english',
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
        print(f"[DEBUG] Target language: {qa_settings.get('target_language', 'NOT SET')}")
        print(f"[DEBUG] Word count check enabled: {qa_settings.get('check_word_count_ratio', False)}")
        
        # Optionally skip mode dialog if a mode override was provided (e.g., scanning phase)
        selected_mode_value = mode_override if mode_override else None
        if selected_mode_value is None:
            # Show mode selection dialog with settings - calculate proportional sizing (halved)
            screen = QApplication.primaryScreen().geometry()
            screen_width = screen.width()
            screen_height = screen.height()
            dialog_width = int(screen_width * 0.45)  # 50% of screen width
            dialog_height = int(screen_height * 0.43)  # 45% of screen height
            
            mode_dialog = QDialog(self)
            mode_dialog.setWindowTitle("Select QA Scanner Mode")
            mode_dialog.resize(dialog_width, dialog_height)
            mode_dialog.setModal(True)
            # Set window icon
            try:
                ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    mode_dialog.setWindowIcon(QIcon(ico_path))
            except Exception:
                pass
            
            # Apply global stylesheet for consistent appearance
            mode_dialog.setStyleSheet("""
                QDialog {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1a1a2e, stop:1 #16213e);
                }
                QPushButton {
                    border: 1px solid #4a5568;
                    border-radius: 4px;
                    padding: 8px 16px;
                    background-color: #2d3748;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #4a5568;
                    border-color: #718096;
                }
                QPushButton:pressed {
                    background-color: #1a202c;
                }
            """)
        
        if selected_mode_value is None:
            # Set minimum size to prevent dialog from being too small (using ratios)
            # 35% width, 35% height for better content fit
            min_width = int(screen_width * 0.35)
            min_height = int(screen_height * 0.35)
            mode_dialog.setMinimumSize(min_width, min_height)
            
            # Variables
            # selected_mode_value already set above
            
            # Main container with constrained expansion
            main_layout = QVBoxLayout(mode_dialog)
            main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Content widget with padding
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(15, 10, 15, 10)
            main_layout.addWidget(content_widget)
            
            # Title with subtitle
            title_label = QLabel("Select Detection Mode")
            title_label.setFont(QFont("Arial", 20, QFont.Bold))
            title_label.setStyleSheet("color: #f0f0f0;")
            title_label.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(title_label)
            
            subtitle_label = QLabel("Choose how sensitive the duplicate detection should be")
            subtitle_label.setFont(QFont("Arial", 11))
            subtitle_label.setStyleSheet("color: #d0d0d0;")
            subtitle_label.setAlignment(Qt.AlignCenter)
            content_layout.addWidget(subtitle_label)
            content_layout.addSpacing(8)
            
            # Mode cards container
            modes_widget = QWidget()
            modes_layout = QGridLayout(modes_widget)
            modes_layout.setSpacing(8)
            content_layout.addWidget(modes_widget)
                    
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
                "recommendation": "✅ Recommended for average use"
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
                modes_layout.setColumnStretch(col, 1)
            
            for idx, mi in enumerate(mode_data):
                # Main card frame with initial background and border
                card = QFrame()
                card.setFrameShape(QFrame.StyledPanel)
                card.setStyleSheet(f"""
                    QFrame {{
                        background-color: {mi["bg_color"]};
                        border: 2px solid {mi["border_color"]};
                        border-radius: 5px;
                    }}
                    QFrame:hover {{
                        background-color: {mi["hover_color"]};
                    }}
                """)
                card.setCursor(Qt.PointingHandCursor)
                modes_layout.addWidget(card, 0, idx)
                
                # Content layout
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(10, 10, 10, 5)
                
                # Icon/Emoji container with fixed height for alignment
                icon_container = QWidget()
                icon_container.setFixedHeight(60)
                icon_container.setStyleSheet("background-color: transparent;")
                icon_container_layout = QVBoxLayout(icon_container)
                icon_container_layout.setContentsMargins(0, 0, 0, 0)
                icon_container_layout.setAlignment(Qt.AlignCenter)
                
                # Icon/Emoji - use Halgakos.ico for AI Hunter, emoji for others
                if mi["value"] == "ai-hunter":
                    # Use Halgakos icon for AI Hunter
                    try:
                        ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
                        if os.path.isfile(ico_path):
                            icon_label = QLabel()
                            # Load icon from QIcon to get best size, then convert to pixmap
                            icon = QIcon(ico_path)
                            # Get the available sizes and pick closest to desired size
                            available_sizes = icon.availableSizes()
                            if available_sizes:
                                # Find size closest to 56x56
                                target_size = 56
                                best_size = min(available_sizes, 
                                              key=lambda s: abs(s.width() - target_size) + abs(s.height() - target_size))
                                # Get pixmap at native resolution
                                original_pixmap = icon.pixmap(best_size)
                            else:
                                # Fallback if no sizes available
                                original_pixmap = QPixmap(ico_path)
                            
                            if not original_pixmap.isNull():
                                # Scale from best native size with high quality
                                scaled_pixmap = original_pixmap.scaled(
                                    56, 56,
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation
                                )
                                icon_label.setPixmap(scaled_pixmap)
                                icon_label.setAlignment(Qt.AlignCenter)
                                icon_label.setStyleSheet("background-color: transparent; border: none;")
                                icon_container_layout.addWidget(icon_label)
                        else:
                            # Fallback to emoji if icon not found
                            emoji_label = QLabel(mi["emoji"])
                            emoji_label.setFont(QFont("Arial", 38))
                            emoji_label.setAlignment(Qt.AlignCenter)
                            emoji_label.setStyleSheet("background-color: transparent; color: white; border: none;")
                            icon_container_layout.addWidget(emoji_label)
                    except Exception:
                        # Fallback to emoji if error
                        emoji_label = QLabel(mi["emoji"])
                        emoji_label.setFont(QFont("Arial", 38))
                        emoji_label.setAlignment(Qt.AlignCenter)
                        emoji_label.setStyleSheet("background-color: transparent; color: white; border: none;")
                        icon_container_layout.addWidget(emoji_label)
                else:
                    # Use emoji for other cards
                    emoji_label = QLabel(mi["emoji"])
                    emoji_label.setFont(QFont("Arial", 38))
                    emoji_label.setAlignment(Qt.AlignCenter)
                    emoji_label.setStyleSheet("background-color: transparent; color: white; border: none;")
                    icon_container_layout.addWidget(emoji_label)
                
                card_layout.addWidget(icon_container)
                
                # Title
                title_label = QLabel(mi["title"])
                title_label.setFont(QFont("Arial", 16, QFont.Bold))
                title_label.setAlignment(Qt.AlignCenter)
                title_label.setStyleSheet(f"background-color: transparent; color: white; border: none;")
                card_layout.addWidget(title_label)
                
                # Subtitle
                subtitle_label = QLabel(mi["subtitle"])
                subtitle_label.setFont(QFont("Arial", 10))
                subtitle_label.setAlignment(Qt.AlignCenter)
                subtitle_label.setStyleSheet(f"background-color: transparent; color: {mi['accent_color']}; border: none;")
                card_layout.addWidget(subtitle_label)
                card_layout.addSpacing(6)
                
                # Features
                for feature in mi["features"]:
                    feature_label = QLabel(feature)
                    feature_label.setFont(QFont("Arial", 9))
                    feature_label.setStyleSheet(f"background-color: transparent; color: #e0e0e0; border: none;")
                    card_layout.addWidget(feature_label)
                
                # Recommendation badge if present
                if mi["recommendation"]:
                    card_layout.addSpacing(6)
                    rec_label = QLabel(mi["recommendation"])
                    rec_label.setFont(QFont("Arial", 9, QFont.Bold))
                    rec_label.setStyleSheet(f"""
                        background-color: {mi['accent_color']};
                        color: white;
                        padding: 3px 6px;
                        border-radius: 3px;
                    """)
                    rec_label.setAlignment(Qt.AlignCenter)
                    card_layout.addWidget(rec_label)
                
                card_layout.addStretch()
                
                # Click handler
                def make_click_handler(mode_value):
                    def handler():
                        nonlocal selected_mode_value
                        selected_mode_value = mode_value
                        mode_dialog.accept()
                    return handler
                
                # Make card clickable with mouse press event
                card.mousePressEvent = lambda event, handler=make_click_handler(mi["value"]): handler()
        
        if selected_mode_value is None:
            # Add separator line before buttons
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setStyleSheet("background-color: #cccccc;")
            separator.setFixedHeight(1)
            content_layout.addWidget(separator)
            content_layout.addSpacing(10)
            
            # Add settings button layout
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            def show_qa_settings():
                """Show QA Scanner settings dialog"""
                self.show_qa_scanner_settings(mode_dialog, qa_settings)
            
            # Auto-search checkbox
            if not hasattr(self, 'qa_auto_search_output_checkbox'):
                self.qa_auto_search_output_checkbox = self._create_styled_checkbox("Auto-search output")
                self.qa_auto_search_output_checkbox.setChecked(self.config.get('qa_auto_search_output', True))
            button_layout.addWidget(self.qa_auto_search_output_checkbox)
            button_layout.addSpacing(10)
            
            settings_btn = QPushButton("⚙️  Scanner Settings")
            settings_btn.setMinimumWidth(140)
            settings_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0d6efd;
                    color: white;
                    border: 1px solid #0d6efd;
                    padding: 8px 10px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #0b5ed7;
                }
            """)
            settings_btn.clicked.connect(show_qa_settings)
            button_layout.addWidget(settings_btn)
            button_layout.addSpacing(10)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setMinimumWidth(100)
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: 1px solid #dc3545;
                    padding: 8px 10px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #bb2d3b;
                }
            """)
            cancel_btn.clicked.connect(mode_dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            button_layout.addStretch()
            content_layout.addLayout(button_layout)
            
            # Handle window close (X button)
            def on_close():
                nonlocal selected_mode_value
                selected_mode_value = None
            mode_dialog.rejected.connect(on_close)
            
            # Show dialog and wait for result
            result = mode_dialog.exec()
            
            # Check if user canceled or selected a mode
            if result == QDialog.Rejected or selected_mode_value is None:
                self.append_log("⚠️ QA scan canceled.")
                return

        # End of optional mode dialog
        
        # Show custom settings dialog if custom mode is selected
        # BUT skip the dialog if non_interactive=True (e.g., post-translation scan)
        if selected_mode_value == "custom" and not non_interactive:
            # Create custom settings dialog
            custom_dialog = QDialog(self)
            custom_dialog.setWindowTitle("Custom Mode Settings")
            custom_dialog.setModal(True)
            # Use screen ratios: 20% width, 50% height for better content fit
            screen = QApplication.primaryScreen().geometry()
            custom_width = int(screen.width() * 0.41)
            custom_height = int(screen.height() * 0.60)
            custom_dialog.resize(custom_width, custom_height)
            # Set window icon
            try:
                ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
                if os.path.isfile(ico_path):
                    custom_dialog.setWindowIcon(QIcon(ico_path))
            except Exception:
                pass
            
            # Main layout
            dialog_layout = QVBoxLayout(custom_dialog)
            
            # Scroll area
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Scrollable content widget
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            scroll.setWidget(scroll_widget)
            dialog_layout.addWidget(scroll)
            
            # Variables for custom settings (using native Python values instead of tk vars)
            custom_settings = {
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
            
            # Store widget references
            custom_widgets = {}
            
            # Title using consistent styling
            title_label = QLabel("Configure Custom Detection Settings")
            title_label.setFont(QFont('Arial', 20, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            scroll_layout.addWidget(title_label)
            scroll_layout.addSpacing(20)
            
            # Detection Thresholds Section
            threshold_group = QGroupBox("Detection Thresholds (%)")
            threshold_group.setFont(QFont('Arial', 12, QFont.Bold))
            threshold_layout = QVBoxLayout(threshold_group)
            threshold_layout.setContentsMargins(25, 25, 25, 25)
            scroll_layout.addWidget(threshold_group)
            
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
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 8, 0, 8)
                
                # Left side - labels
                label_widget = QWidget()
                label_layout = QVBoxLayout(label_widget)
                label_layout.setContentsMargins(0, 0, 0, 0)
                
                main_label = QLabel(f"{label_text} - {description}:")
                main_label.setFont(QFont('Arial', 11))
                label_layout.addWidget(main_label)
                label_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                row_layout.addWidget(label_widget)
                
                # Right side - slider and percentage
                slider_widget = QWidget()
                slider_layout = QHBoxLayout(slider_widget)
                slider_layout.setContentsMargins(20, 0, 0, 0)
                
                # Create slider
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(10)
                slider.setMaximum(100)
                slider.setValue(custom_settings[setting_key])
                slider.setMinimumWidth(300)
                # Disable mousewheel scrolling on slider
                slider.wheelEvent = lambda event: event.ignore()
                slider_layout.addWidget(slider)
                
                # Percentage label (shows current value)
                percentage_label = QLabel(f"{custom_settings[setting_key]}%")
                percentage_label.setFont(QFont('Arial', 12, QFont.Bold))
                percentage_label.setMinimumWidth(50)
                percentage_label.setAlignment(Qt.AlignRight)
                slider_layout.addWidget(percentage_label)
                percentage_labels[setting_key] = percentage_label
                
                row_layout.addWidget(slider_widget)
                threshold_layout.addWidget(row_widget)
                
                # Store slider widget reference
                custom_widgets[setting_key] = slider
                
                # Update percentage label when slider moves
                def create_update_function(key, label, settings_dict):
                    def update_percentage(value):
                        settings_dict[key] = value
                        label.setText(f"{value}%")
                    return update_percentage
                
                # Connect slider to update function
                update_func = create_update_function(setting_key, percentage_label, custom_settings)
                slider.valueChanged.connect(update_func)
            
            scroll_layout.addSpacing(15)
            
            # Processing Options Section
            options_group = QGroupBox("Processing Options")
            options_group.setFont(QFont('Arial', 12, QFont.Bold))
            options_layout = QVBoxLayout(options_group)
            options_layout.setContentsMargins(20, 20, 20, 20)
            scroll_layout.addWidget(options_group)
            
            # Consecutive chapters option with spinbox
            consec_widget = QWidget()
            consec_layout = QHBoxLayout(consec_widget)
            consec_layout.setContentsMargins(0, 5, 0, 5)
            
            consec_label = QLabel("Consecutive chapters to check:")
            consec_label.setFont(QFont('Arial', 11))
            consec_layout.addWidget(consec_label)
            
            consec_spinbox = QSpinBox()
            consec_spinbox.setMinimum(1)
            consec_spinbox.setMaximum(10)
            consec_spinbox.setValue(custom_settings['consecutive_chapters'])
            consec_spinbox.setMinimumWidth(100)
            # Disable mousewheel scrolling
            consec_spinbox.wheelEvent = lambda event: event.ignore()
            consec_layout.addWidget(consec_spinbox)
            consec_layout.addStretch()
            options_layout.addWidget(consec_widget)
            custom_widgets['consecutive_chapters'] = consec_spinbox
            
            # Sample size option
            sample_widget = QWidget()
            sample_layout = QHBoxLayout(sample_widget)
            sample_layout.setContentsMargins(0, 5, 0, 5)
            
            sample_label = QLabel("Sample size for comparison (characters):")
            sample_label.setFont(QFont('Arial', 11))
            sample_layout.addWidget(sample_label)
            
            # Sample size spinbox with larger range
            sample_spinbox = QSpinBox()
            sample_spinbox.setMinimum(1000)
            sample_spinbox.setMaximum(10000)
            sample_spinbox.setSingleStep(500)
            sample_spinbox.setValue(custom_settings['sample_size'])
            sample_spinbox.setMinimumWidth(100)
            # Disable mousewheel scrolling
            sample_spinbox.wheelEvent = lambda event: event.ignore()
            sample_layout.addWidget(sample_spinbox)
            sample_layout.addStretch()
            options_layout.addWidget(sample_widget)
            custom_widgets['sample_size'] = sample_spinbox
            
            # Minimum text length option
            min_length_widget = QWidget()
            min_length_layout = QHBoxLayout(min_length_widget)
            min_length_layout.setContentsMargins(0, 5, 0, 5)
            
            min_length_label = QLabel("Minimum text length to process (characters):")
            min_length_label.setFont(QFont('Arial', 11))
            min_length_layout.addWidget(min_length_label)
            
            # Minimum length spinbox
            min_length_spinbox = QSpinBox()
            min_length_spinbox.setMinimum(100)
            min_length_spinbox.setMaximum(5000)
            min_length_spinbox.setSingleStep(100)
            min_length_spinbox.setValue(custom_settings['min_text_length'])
            min_length_spinbox.setMinimumWidth(100)
            # Disable mousewheel scrolling
            min_length_spinbox.wheelEvent = lambda event: event.ignore()
            min_length_layout.addWidget(min_length_spinbox)
            min_length_layout.addStretch()
            options_layout.addWidget(min_length_widget)
            custom_widgets['min_text_length'] = min_length_spinbox
            
            # Check all file pairs option
            check_all_checkbox = self._create_styled_checkbox("Check all file pairs (slower but more thorough)")
            check_all_checkbox.setChecked(custom_settings['check_all_pairs'])
            options_layout.addWidget(check_all_checkbox)
            custom_widgets['check_all_pairs'] = check_all_checkbox
            
            scroll_layout.addSpacing(30)
            
            # Create button layout at bottom
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.addStretch()
            scroll_layout.addWidget(button_widget)
            
            # Flag to track if settings were saved
            settings_saved = False
            
            def save_custom_settings():
                """Save custom settings and close dialog"""
                nonlocal settings_saved
                qa_settings['custom_mode_settings'] = {
                    'thresholds': {
                        'similarity': custom_widgets['similarity'].value() / 100,
                        'semantic': custom_widgets['semantic'].value() / 100,
                        'structural': custom_widgets['structural'].value() / 100,
                        'word_overlap': custom_widgets['word_overlap'].value() / 100,
                        'minhash_threshold': custom_widgets['minhash_threshold'].value() / 100
                    },
                    'consecutive_chapters': custom_widgets['consecutive_chapters'].value(),
                    'check_all_pairs': custom_widgets['check_all_pairs'].isChecked(),
                    'sample_size': custom_widgets['sample_size'].value(),
                    'min_text_length': custom_widgets['min_text_length'].value()
                }
                settings_saved = True
                self.append_log("✅ Custom detection settings saved")
                custom_dialog.accept()
            
            def reset_to_defaults():
                """Reset all values to default settings"""
                reply = QMessageBox.question(custom_dialog, "Reset to Defaults", 
                                           "Reset all values to default settings?",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    custom_widgets['similarity'].setValue(85)
                    custom_widgets['semantic'].setValue(80)
                    custom_widgets['structural'].setValue(90)
                    custom_widgets['word_overlap'].setValue(75)
                    custom_widgets['minhash_threshold'].setValue(80)
                    custom_widgets['consecutive_chapters'].setValue(2)
                    custom_widgets['check_all_pairs'].setChecked(False)
                    custom_widgets['sample_size'].setValue(3000)
                    custom_widgets['min_text_length'].setValue(500)
                    self.append_log("ℹ️ Settings reset to defaults")
            
            # Flag to prevent recursive cancel calls
            cancel_in_progress = False
            
            def cancel_settings():
                """Cancel without saving"""
                nonlocal settings_saved, cancel_in_progress
                
                # Prevent recursive calls
                if cancel_in_progress:
                    return
                    
                cancel_in_progress = True
                try:
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
                            if key == 'check_all_pairs':
                                if custom_widgets[key].isChecked() != default_val:
                                    changed = True
                                    break
                            else:
                                if custom_widgets[key].value() != default_val:
                                    changed = True
                                    break
                        
                        if changed:
                            reply = QMessageBox.question(custom_dialog, "Unsaved Changes", 
                                                        "You have unsaved changes. Are you sure you want to cancel?",
                                                        QMessageBox.Yes | QMessageBox.No)
                            if reply == QMessageBox.Yes:
                                # Disconnect signal before rejecting to prevent loop
                                try:
                                    custom_dialog.rejected.disconnect(cancel_settings)
                                except:
                                    pass
                                custom_dialog.reject()
                        else:
                            # Disconnect signal before rejecting to prevent loop
                            try:
                                custom_dialog.rejected.disconnect(cancel_settings)
                            except:
                                pass
                            custom_dialog.reject()
                    else:
                        # Disconnect signal before rejecting to prevent loop
                        try:
                            custom_dialog.rejected.disconnect(cancel_settings)
                        except:
                            pass
                        custom_dialog.reject()
                finally:
                    cancel_in_progress = False
            
            # Create buttons
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setMinimumWidth(120)
            cancel_btn.clicked.connect(cancel_settings)
            button_layout.addWidget(cancel_btn)
            
            reset_btn = QPushButton("Reset Defaults")
            reset_btn.setMinimumWidth(120)
            reset_btn.clicked.connect(reset_to_defaults)
            button_layout.addWidget(reset_btn)
            
            start_btn = QPushButton("Start Scan")
            start_btn.setMinimumWidth(120)
            start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: 1px solid #28a745;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            start_btn.clicked.connect(save_custom_settings)
            button_layout.addWidget(start_btn)
            
            button_layout.addStretch()
            
            # Handle window close properly - treat as cancel
            # Store the connection so we can disconnect it later if needed
            rejected_connection = custom_dialog.rejected.connect(cancel_settings)
            
            # Show dialog and wait for result
            result = custom_dialog.exec()
            
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
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("No Source EPUB Selected")
                msg.setText("Word count cross-reference is enabled but no source EPUB file is selected.")
                msg.setInformativeText("Would you like to:\n"
                                      "• YES - Continue scan without word count analysis\n"
                                      "• NO - Select an EPUB file now\n"
                                      "• CANCEL - Cancel the scan")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                result = msg.exec()
                
                if result == QMessageBox.Cancel:
                    self.append_log("⚠️ QA scan canceled.")
                    return
                elif result == QMessageBox.No:  # No - Select EPUB now
                    epub_path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select Source EPUB File",
                        "",
                        "EPUB files (*.epub);;All files (*.*)"
                    )
                    
                    if not epub_path:
                        retry = QMessageBox.question(
                            self,
                            "No File Selected",
                            "No EPUB file was selected.\n\n" +
                            "Do you want to continue the scan without word count analysis?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        
                        if retry == QMessageBox.No:
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
            self.config['qa_auto_search_output'] = bool(self.qa_auto_search_output_checkbox.isChecked())
            self.save_config(show_message=False)
        except Exception:
            pass
        
        # Try to auto-detect output folders based on EPUB files
        folders_to_scan = []
        auto_search_enabled = self.config.get('qa_auto_search_output', True)
        try:
            if hasattr(self, 'qa_auto_search_output_checkbox'):
                auto_search_enabled = bool(self.qa_auto_search_output_checkbox.isChecked())
        except Exception:
            pass
        
        # Debug output for scanning phase removed
        
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
                
                selected_folder = QFileDialog.getExistingDirectory(
                    self.parent,
                    "Auto-search failed - Select Output Folder to Scan"
                )
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
            selected_folder = QFileDialog.getExistingDirectory(
                self.parent,
                "Select Folder with HTML Files"
            )
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
                                msg = QMessageBox(self)
                                msg.setIcon(QMessageBox.Warning)
                                msg.setWindowTitle("EPUB/Folder Name Mismatch")
                                msg.setText(f"The source EPUB and output folder names don't match:\n\n"
                                          f"📖 EPUB: {epub_name}\n"
                                          f"📁 Folder: {folder_name_for_check}\n\n"
                                          "This might mean you're comparing the wrong files.")
                                msg.setInformativeText("Would you like to:\n"
                                                      "• YES - Continue anyway (I'm sure these match)\n"
                                                      "• NO - Select a different EPUB file\n"
                                                      "• CANCEL - Cancel the scan")
                                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                                result = msg.exec()
                                
                                if result == QMessageBox.Cancel:
                                    self.append_log("⚠️ QA scan canceled due to EPUB/folder mismatch.")
                                    return
                                elif result == QMessageBox.No:  # No - select different EPUB
                                    new_epub_path, _ = QFileDialog.getOpenFileName(
                                        self,
                                        "Select Different Source EPUB File",
                                        "",
                                        "EPUB files (*.epub);;All files (*.*)"
                                    )
                                    
                                    if new_epub_path:
                                        current_epub_path = new_epub_path
                                        self.selected_epub_path = new_epub_path
                                        self.config['last_epub_path'] = new_epub_path
                                        self.save_config(show_message=False)
                                        self.append_log(f"✅ Updated EPUB: {os.path.basename(new_epub_path)}")
                                    else:
                                        proceed = QMessageBox.question(
                                            self,
                                            "No File Selected",
                                            "No EPUB file was selected.\n\n" +
                                            "Continue scan without word count analysis?",
                                            QMessageBox.Yes | QMessageBox.No
                                        )
                                        if proceed == QMessageBox.No:
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
                # Emit signal to update button (thread-safe)
                self.thread_complete_signal.emit()
        
        # Run via shared executor
        self._ensure_executor()
        if self.executor:
            self.qa_future = self.executor.submit(run_scan)
            # Ensure UI is refreshed when QA work completes (button update handled by thread_complete_signal in finally block)
            def _qa_done_callback(f):
                try:
                    self.qa_future = None
                except Exception:
                    pass
            try:
                self.qa_future.add_done_callback(_qa_done_callback)
            except Exception:
                pass
        else:
            self.qa_thread = threading.Thread(target=run_scan, daemon=True)
            self.qa_thread.start()
        
        # Update button IMMEDIATELY after starting thread (synchronous)
        self.update_run_button()

    def show_qa_scanner_settings(self, parent_dialog, qa_settings):
        """Show QA Scanner settings dialog"""
        # Create settings dialog
        dialog = QDialog(parent_dialog)
        dialog.setWindowTitle("QA Scanner Settings")
        dialog.setModal(True)
        # Use screen ratios: 40% width, 85% height (decreased from 100%)
        screen = QApplication.primaryScreen().geometry()
        settings_width = int(screen.width() * 0.37)
        settings_height = int(screen.height() * 0.85)
        dialog.resize(settings_width, settings_height)
        
        # Set window icon
        try:
            ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                dialog.setWindowIcon(QIcon(ico_path))
        except Exception:
            pass
        
        # Apply basic dark stylesheet
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QGroupBox {
                color: white;
                border: 1px solid #555;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: white;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QComboBox {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                padding-right: 25px;
            }
            QComboBox:hover {
                background-color: #505050;
                border: 1px solid #777;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #555;
            }
            QComboBox::down-arrow {
                image: url(Halgakos.ico);
                width: 16px;
                height: 16px;
            }
            QComboBox:on {
                border: 1px solid #888;
            }
            QComboBox QAbstractItemView {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
                selection-background-color: #505050;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Scrollable content widget
        scroll_widget = QWidget()
        scroll_widget.setObjectName('scroll_widget')
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(30, 20, 30, 20)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # Helper function to disable mousewheel on spinboxes and comboboxes
        def disable_wheel_event(widget):
            widget.wheelEvent = lambda event: event.ignore()
        
        # Title
        title_label = QLabel("QA Scanner Settings")
        title_label.setFont(QFont('Arial', 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(title_label)
        scroll_layout.addSpacing(20)
        
        # Foreign Character Settings Section
        foreign_group = QGroupBox("Foreign Character Detection")
        foreign_group.setFont(QFont('Arial', 12, QFont.Bold))
        foreign_layout = QVBoxLayout(foreign_group)
        foreign_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(foreign_group)
        
        # Target Language setting
        target_lang_widget = QWidget()
        target_lang_layout = QHBoxLayout(target_lang_widget)
        target_lang_layout.setContentsMargins(0, 0, 0, 10)
        
        target_lang_label = QLabel("Target language:")
        target_lang_label.setFont(QFont('Arial', 10))
        target_lang_layout.addWidget(target_lang_label)
        
        # Capitalize the stored value for display in combobox
        stored_language = qa_settings.get('target_language', 'english')
        display_language = stored_language.capitalize()
        target_language_options = [
            'English', 'Spanish', 'French', 'German', 'Portuguese', 
            'Italian', 'Russian', 'Japanese', 'Korean', 'Chinese', 
            'Arabic', 'Hebrew', 'Thai'
        ]
        
        target_language_combo = QComboBox()
        target_language_combo.addItems(target_language_options)
        target_language_combo.setCurrentText(display_language)
        target_language_combo.setMinimumWidth(150)
        disable_wheel_event(target_language_combo)
        target_lang_layout.addWidget(target_language_combo)
        
        target_lang_hint = QLabel("(characters from other scripts will be flagged)")
        target_lang_hint.setFont(QFont('Arial', 9))
        target_lang_hint.setStyleSheet("color: gray;")
        target_lang_layout.addWidget(target_lang_hint)
        target_lang_layout.addStretch()
        foreign_layout.addWidget(target_lang_widget)
        
        # Threshold setting
        threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(0, 10, 0, 10)
        
        threshold_label = QLabel("Minimum foreign characters to flag:")
        threshold_label.setFont(QFont('Arial', 10))
        threshold_layout.addWidget(threshold_label)
        
        threshold_spinbox = QSpinBox()
        threshold_spinbox.setMinimum(0)
        threshold_spinbox.setMaximum(1000)
        threshold_spinbox.setValue(qa_settings.get('foreign_char_threshold', 10))
        threshold_spinbox.setMinimumWidth(100)
        disable_wheel_event(threshold_spinbox)
        threshold_layout.addWidget(threshold_spinbox)
        
        threshold_hint = QLabel("(0 = always flag, higher = more tolerant)")
        threshold_hint.setFont(QFont('Arial', 9))
        threshold_hint.setStyleSheet("color: gray;")
        threshold_layout.addWidget(threshold_hint)
        threshold_layout.addStretch()
        foreign_layout.addWidget(threshold_widget)
        
        # Excluded characters
        excluded_label = QLabel("Additional characters to exclude from detection:")
        excluded_label.setFont(QFont('Arial', 10))
        foreign_layout.addWidget(excluded_label)
        
        # Text edit for excluded characters
        excluded_text = QTextEdit()
        excluded_text.setMaximumHeight(150)
        excluded_text.setFont(QFont('Consolas', 10))
        excluded_text.setPlainText(qa_settings.get('excluded_characters', ''))
        foreign_layout.addWidget(excluded_text)
        
        excluded_hint = QLabel("Enter characters separated by spaces (e.g., ™ © ® • …)")
        excluded_hint.setFont(QFont('Arial', 9))
        excluded_hint.setStyleSheet("color: gray;")
        foreign_layout.addWidget(excluded_hint)
        
        scroll_layout.addSpacing(20)
        
        # Detection Options Section
        detection_group = QGroupBox("Detection Options")
        detection_group.setFont(QFont('Arial', 12, QFont.Bold))
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(detection_group)
        
        # Checkboxes for detection options
        check_encoding_checkbox = self._create_styled_checkbox("Check for encoding issues (�, □, ◇)")
        check_encoding_checkbox.setChecked(qa_settings.get('check_encoding_issues', False))
        detection_layout.addWidget(check_encoding_checkbox)
        
        check_repetition_checkbox = self._create_styled_checkbox("Check for excessive repetition")
        check_repetition_checkbox.setChecked(qa_settings.get('check_repetition', True))
        detection_layout.addWidget(check_repetition_checkbox)
        
        check_artifacts_checkbox = self._create_styled_checkbox("Check for translation artifacts (MTL notes, watermarks)")
        check_artifacts_checkbox.setChecked(qa_settings.get('check_translation_artifacts', False))
        detection_layout.addWidget(check_artifacts_checkbox)
        
        check_glossary_checkbox = self._create_styled_checkbox("Check for glossary leakage (raw glossary entries in translation)")
        check_glossary_checkbox.setChecked(qa_settings.get('check_glossary_leakage', True))
        detection_layout.addWidget(check_glossary_checkbox)
        
        scroll_layout.addSpacing(20)
        
        # File Processing Section
        file_group = QGroupBox("File Processing")
        file_group.setFont(QFont('Arial', 12, QFont.Bold))
        file_layout = QVBoxLayout(file_group)
        file_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(file_group)
        
        # Minimum file length
        min_length_widget = QWidget()
        min_length_layout = QHBoxLayout(min_length_widget)
        min_length_layout.setContentsMargins(0, 0, 0, 10)
        
        min_length_label = QLabel("Minimum file length (characters):")
        min_length_label.setFont(QFont('Arial', 10))
        min_length_layout.addWidget(min_length_label)
        
        min_length_spinbox = QSpinBox()
        min_length_spinbox.setMinimum(0)
        min_length_spinbox.setMaximum(10000)
        min_length_spinbox.setValue(qa_settings.get('min_file_length', 0))
        min_length_spinbox.setMinimumWidth(100)
        disable_wheel_event(min_length_spinbox)
        min_length_layout.addWidget(min_length_spinbox)
        min_length_layout.addStretch()
        file_layout.addWidget(min_length_widget)

        scroll_layout.addSpacing(15)
        
        # Word Count Cross-Reference Section
        wordcount_group = QGroupBox("Word Count Analysis")
        wordcount_group.setFont(QFont('Arial', 12, QFont.Bold))
        wordcount_layout = QVBoxLayout(wordcount_group)
        wordcount_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(wordcount_group)
        
        check_word_count_checkbox = self._create_styled_checkbox("Cross-reference word counts with original EPUB")
        check_word_count_checkbox.setChecked(qa_settings.get('check_word_count_ratio', False))
        wordcount_layout.addWidget(check_word_count_checkbox)
        
        wordcount_desc = QLabel("Compares word counts between original and translated files to detect missing content.\n" +
                               "Accounts for typical expansion ratios when translating from CJK to English.")
        wordcount_desc.setFont(QFont('Arial', 9))
        wordcount_desc.setStyleSheet("color: gray;")
        wordcount_desc.setWordWrap(True)
        wordcount_desc.setMaximumWidth(700)
        wordcount_layout.addWidget(wordcount_desc)
 
        # Show current EPUB status and allow selection
        epub_widget = QWidget()
        epub_layout = QHBoxLayout(epub_widget)
        epub_layout.setContentsMargins(0, 10, 0, 5)

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

        status_label = QLabel(status_text)
        status_label.setFont(QFont('Arial', 10))
        status_label.setStyleSheet(f"color: {status_color};")
        epub_layout.addWidget(status_label)

        def select_epub_for_qa():
            epub_path, _ = QFileDialog.getOpenFileName(
                dialog,
                "Select Source EPUB File",
                "",
                "EPUB files (*.epub);;All files (*.*)"
            )
            if epub_path:
                self.selected_epub_path = epub_path
                self.config['last_epub_path'] = epub_path
                self.save_config(show_message=False)
                
                # Clear multiple EPUB tracking when manually selecting a single EPUB
                if hasattr(self, 'selected_epub_files'):
                    self.selected_epub_files = [epub_path]
                
                status_label.setText(f"📖 Current EPUB: {os.path.basename(epub_path)}")
                status_label.setStyleSheet("color: green;")
                self.append_log(f"✅ Selected EPUB for QA: {os.path.basename(epub_path)}")

        select_epub_btn = QPushButton("Select EPUB")
        select_epub_btn.setFont(QFont('Arial', 9))
        select_epub_btn.clicked.connect(select_epub_for_qa)
        epub_layout.addWidget(select_epub_btn)
        epub_layout.addStretch()
        wordcount_layout.addWidget(epub_widget)

        # Add option to disable mismatch warning
        warn_mismatch_checkbox = self._create_styled_checkbox("Warn when EPUB and folder names don't match")
        warn_mismatch_checkbox.setChecked(qa_settings.get('warn_name_mismatch', True))
        wordcount_layout.addWidget(warn_mismatch_checkbox)

        scroll_layout.addSpacing(20)
        
        # Additional Checks Section
        additional_group = QGroupBox("Additional Checks")
        additional_group.setFont(QFont('Arial', 12, QFont.Bold))
        additional_layout = QVBoxLayout(additional_group)
        additional_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(additional_group)

        # Multiple headers check
        check_multiple_headers_checkbox = self._create_styled_checkbox("Detect files with 2 or more headers (h1-h6 tags)")
        check_multiple_headers_checkbox.setChecked(qa_settings.get('check_multiple_headers', True))
        additional_layout.addWidget(check_multiple_headers_checkbox)

        headers_desc = QLabel("Identifies files that may have been incorrectly split or merged.\n" +
                             "Useful for detecting chapters that contain multiple sections.")
        headers_desc.setFont(QFont('Arial', 9))
        headers_desc.setStyleSheet("color: gray;")
        headers_desc.setWordWrap(True)
        headers_desc.setMaximumWidth(700)
        additional_layout.addWidget(headers_desc)
        additional_layout.addSpacing(10)

        # Missing HTML tag check
        html_tag_widget = QWidget()
        html_tag_layout = QHBoxLayout(html_tag_widget)
        html_tag_layout.setContentsMargins(0, 0, 0, 5)

        check_missing_html_tag_checkbox = self._create_styled_checkbox("Flag HTML files with missing <html> tag")
        check_missing_html_tag_checkbox.setChecked(qa_settings.get('check_missing_html_tag', True))
        html_tag_layout.addWidget(check_missing_html_tag_checkbox)

        html_tag_hint = QLabel("(Checks if HTML files have proper structure)")
        html_tag_hint.setFont(QFont('Arial', 9))
        html_tag_hint.setStyleSheet("color: gray;")
        html_tag_layout.addWidget(html_tag_hint)
        html_tag_layout.addStretch()
        additional_layout.addWidget(html_tag_widget)

        # Invalid nesting check (separate toggle)
        check_invalid_nesting_checkbox = self._create_styled_checkbox("Check for invalid tag nesting")
        check_invalid_nesting_checkbox.setChecked(qa_settings.get('check_invalid_nesting', False))
        additional_layout.addWidget(check_invalid_nesting_checkbox)

        additional_layout.addSpacing(15)
        
        # NEW: Paragraph Structure Check
        # Separator line
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.HLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        additional_layout.addWidget(separator_line)
        additional_layout.addSpacing(10)
        
        # Checkbox for paragraph structure check
        check_paragraph_structure_checkbox = self._create_styled_checkbox("Check for insufficient paragraph tags")
        check_paragraph_structure_checkbox.setChecked(qa_settings.get('check_paragraph_structure', True))
        additional_layout.addWidget(check_paragraph_structure_checkbox)
        
        # Threshold setting frame
        threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(20, 10, 0, 5)
        
        threshold_label = QLabel("Minimum text in <p> tags:")
        threshold_label.setFont(QFont('Arial', 10))
        threshold_layout.addWidget(threshold_label)
        
        # Get current threshold value (default 30%)
        current_threshold = int(qa_settings.get('paragraph_threshold', 0.3) * 100)
        
        # Spinbox for threshold
        paragraph_threshold_spinbox = QSpinBox()
        paragraph_threshold_spinbox.setMinimum(0)
        paragraph_threshold_spinbox.setMaximum(100)
        paragraph_threshold_spinbox.setValue(current_threshold)
        paragraph_threshold_spinbox.setMinimumWidth(80)
        disable_wheel_event(paragraph_threshold_spinbox)
        threshold_layout.addWidget(paragraph_threshold_spinbox)
        
        percent_label = QLabel("%")
        percent_label.setFont(QFont('Arial', 10))
        threshold_layout.addWidget(percent_label)
        
        # Threshold value label
        threshold_value_label = QLabel(f"(currently {current_threshold}%)")
        threshold_value_label.setFont(QFont('Arial', 9))
        threshold_value_label.setStyleSheet("color: gray;")
        threshold_layout.addWidget(threshold_value_label)
        threshold_layout.addStretch()
        additional_layout.addWidget(threshold_widget)
        
        # Update label when spinbox changes
        def update_threshold_label(value):
            threshold_value_label.setText(f"(currently {value}%)")
        paragraph_threshold_spinbox.valueChanged.connect(update_threshold_label)
        
        # Description
        para_desc = QLabel("Detects HTML files where text content is not properly wrapped in paragraph tags.\n" +
                          "Files with less than the specified percentage of text in <p> tags will be flagged.\n" +
                          "Also checks for large blocks of unwrapped text directly in the body element.")
        para_desc.setFont(QFont('Arial', 9))
        para_desc.setStyleSheet("color: gray;")
        para_desc.setWordWrap(True)
        para_desc.setMaximumWidth(700)
        para_desc.setContentsMargins(20, 5, 0, 0)
        additional_layout.addWidget(para_desc)
        
        # Enable/disable threshold setting based on checkbox
        def toggle_paragraph_threshold(checked):
            paragraph_threshold_spinbox.setEnabled(checked)
            threshold_label.setEnabled(checked)
            percent_label.setEnabled(checked)
            threshold_value_label.setEnabled(checked)
        
        check_paragraph_structure_checkbox.toggled.connect(toggle_paragraph_threshold)
        toggle_paragraph_threshold(check_paragraph_structure_checkbox.isChecked())  # Set initial state

        scroll_layout.addSpacing(20)
        
        # Report Settings Section
        report_group = QGroupBox("Report Settings")
        report_group.setFont(QFont('Arial', 12, QFont.Bold))
        report_layout = QVBoxLayout(report_group)
        report_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(report_group)
        
        # Report format
        format_widget = QWidget()
        format_layout = QHBoxLayout(format_widget)
        format_layout.setContentsMargins(0, 0, 0, 10)
        
        format_label = QLabel("Report format:")
        format_label.setFont(QFont('Arial', 10))
        format_layout.addWidget(format_label)
        
        current_format_value = qa_settings.get('report_format', 'detailed')
        format_options = [
            ("Summary only", "summary"),
            ("Detailed (recommended)", "detailed"),
            ("Verbose (all data)", "verbose")
        ]
        
        # Create radio buttons for format options
        format_radio_buttons = []
        for idx, (text, value) in enumerate(format_options):
            rb = QRadioButton(text)
            if value == current_format_value:
                rb.setChecked(True)
            format_layout.addWidget(rb)
            format_radio_buttons.append((rb, value))
        
        format_layout.addStretch()
        report_layout.addWidget(format_widget)
        
        # Auto-save report
        auto_save_checkbox = self._create_styled_checkbox("Automatically save report after scan")
        auto_save_checkbox.setChecked(qa_settings.get('auto_save_report', True))
        report_layout.addWidget(auto_save_checkbox)

        scroll_layout.addSpacing(20)
        
        # Cache Settings Section
        cache_group = QGroupBox("Performance Cache Settings")
        cache_group.setFont(QFont('Arial', 12, QFont.Bold))
        cache_layout = QVBoxLayout(cache_group)
        cache_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(cache_group)
        
        # Enable cache checkbox
        cache_enabled_checkbox = self._create_styled_checkbox("Enable performance cache (speeds up duplicate detection)")
        cache_enabled_checkbox.setChecked(qa_settings.get('cache_enabled', True))
        cache_layout.addWidget(cache_enabled_checkbox)
        cache_layout.addSpacing(10)
        
        # Cache size settings
        cache_desc_label = QLabel("Cache sizes (0 = disabled, -1 = unlimited):")
        cache_desc_label.setFont(QFont('Arial', 10))
        cache_layout.addWidget(cache_desc_label)
        cache_layout.addSpacing(5)
        
        # Cache size variables - store spinboxes and buttons
        cache_spinboxes = {}
        cache_buttons = {}
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
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 2, 0, 2)
            
            # Label
            label_text = cache_name.replace('_', ' ').title() + ":"
            cache_label = QLabel(label_text)
            cache_label.setFont(QFont('Arial', 9))
            cache_label.setMinimumWidth(200)
            row_layout.addWidget(cache_label)
            
            # Get current value
            current_value = qa_settings.get(f'cache_{cache_name}', default_value)
            
            # Spinbox
            spinbox = QSpinBox()
            spinbox.setMinimum(-1)
            spinbox.setMaximum(50000)
            spinbox.setValue(current_value)
            spinbox.setMinimumWidth(100)
            disable_wheel_event(spinbox)
            row_layout.addWidget(spinbox)
            cache_spinboxes[cache_name] = spinbox
            
            # Quick preset buttons
            def make_preset_handler(sb, val):
                return lambda: sb.setValue(val)
            
            off_btn = QPushButton("Off")
            off_btn.setFont(QFont('Arial', 8))
            off_btn.setMinimumWidth(40)
            off_btn.clicked.connect(make_preset_handler(spinbox, 0))
            row_layout.addWidget(off_btn)
            
            small_btn = QPushButton("Small")
            small_btn.setFont(QFont('Arial', 8))
            small_btn.setMinimumWidth(50)
            small_btn.clicked.connect(make_preset_handler(spinbox, 1000))
            row_layout.addWidget(small_btn)
            
            medium_btn = QPushButton("Medium")
            medium_btn.setFont(QFont('Arial', 8))
            medium_btn.setMinimumWidth(60)
            medium_btn.clicked.connect(make_preset_handler(spinbox, default_value))
            row_layout.addWidget(medium_btn)
            
            large_btn = QPushButton("Large")
            large_btn.setFont(QFont('Arial', 8))
            large_btn.setMinimumWidth(50)
            large_btn.clicked.connect(make_preset_handler(spinbox, default_value * 2))
            row_layout.addWidget(large_btn)
            
            max_btn = QPushButton("Max")
            max_btn.setFont(QFont('Arial', 8))
            max_btn.setMinimumWidth(40)
            max_btn.clicked.connect(make_preset_handler(spinbox, -1))
            row_layout.addWidget(max_btn)
            
            # Store buttons for enabling/disabling
            cache_buttons[cache_name] = [cache_label, off_btn, small_btn, medium_btn, large_btn, max_btn]
            
            row_layout.addStretch()
            cache_layout.addWidget(row_widget)
        
        # Enable/disable cache size controls based on checkbox
        def toggle_cache_controls(checked):
            for cache_name in cache_defaults.keys():
                spinbox = cache_spinboxes[cache_name]
                spinbox.setEnabled(checked)
                for widget in cache_buttons[cache_name]:
                    widget.setEnabled(checked)
        
        cache_enabled_checkbox.toggled.connect(toggle_cache_controls)
        toggle_cache_controls(cache_enabled_checkbox.isChecked())  # Set initial state
        
        cache_layout.addSpacing(10)
        
        # Auto-size cache option
        auto_size_widget = QWidget()
        auto_size_layout = QHBoxLayout(auto_size_widget)
        auto_size_layout.setContentsMargins(0, 0, 0, 5)
        
        auto_size_checkbox = self._create_styled_checkbox("Auto-size caches based on available RAM")
        auto_size_checkbox.setChecked(qa_settings.get('cache_auto_size', False))
        auto_size_layout.addWidget(auto_size_checkbox)
        
        auto_size_hint = QLabel("(overrides manual settings)")
        auto_size_hint.setFont(QFont('Arial', 9))
        auto_size_hint.setStyleSheet("color: gray;")
        auto_size_layout.addWidget(auto_size_hint)
        auto_size_layout.addStretch()
        cache_layout.addWidget(auto_size_widget)
        
        cache_layout.addSpacing(10)
        
        # Cache statistics display
        show_stats_checkbox = self._create_styled_checkbox("Show cache hit/miss statistics after scan")
        show_stats_checkbox.setChecked(qa_settings.get('cache_show_stats', False))
        cache_layout.addWidget(show_stats_checkbox)
        
        cache_layout.addSpacing(10)
        
        # Info about cache
        cache_info = QLabel("Larger cache sizes use more memory but improve performance for:\n" +
                           "• Large datasets (100+ files)\n" +
                           "• AI Hunter mode (all file pairs compared)\n" +
                           "• Repeated scans of the same folder")
        cache_info.setFont(QFont('Arial', 9))
        cache_info.setStyleSheet("color: gray;")
        cache_info.setWordWrap(True)
        cache_info.setMaximumWidth(700)
        cache_info.setContentsMargins(20, 0, 0, 0)
        cache_layout.addWidget(cache_info)

        scroll_layout.addSpacing(20)
        
        # AI Hunter Performance Section
        ai_hunter_group = QGroupBox("AI Hunter Performance Settings")
        ai_hunter_group.setFont(QFont('Arial', 12, QFont.Bold))
        ai_hunter_layout = QVBoxLayout(ai_hunter_group)
        ai_hunter_layout.setContentsMargins(20, 15, 20, 15)
        scroll_layout.addWidget(ai_hunter_group)

        # Description
        ai_hunter_desc = QLabel("AI Hunter mode performs exhaustive duplicate detection by comparing every file pair.\n" +
                               "Parallel processing can significantly speed up this process on multi-core systems.")
        ai_hunter_desc.setFont(QFont('Arial', 9))
        ai_hunter_desc.setStyleSheet("color: gray;")
        ai_hunter_desc.setWordWrap(True)
        ai_hunter_desc.setMaximumWidth(700)
        ai_hunter_layout.addWidget(ai_hunter_desc)
        ai_hunter_layout.addSpacing(10)

        # Parallel workers setting
        workers_widget = QWidget()
        workers_layout = QHBoxLayout(workers_widget)
        workers_layout.setContentsMargins(0, 0, 0, 10)

        workers_label = QLabel("Maximum parallel workers:")
        workers_label.setFont(QFont('Arial', 10))
        workers_layout.addWidget(workers_label)

        # Get current value from AI Hunter config
        ai_hunter_config = self.config.get('ai_hunter_config', {})
        current_max_workers = ai_hunter_config.get('ai_hunter_max_workers', 1)

        ai_hunter_workers_spinbox = QSpinBox()
        ai_hunter_workers_spinbox.setMinimum(0)
        ai_hunter_workers_spinbox.setMaximum(64)
        ai_hunter_workers_spinbox.setValue(current_max_workers)
        ai_hunter_workers_spinbox.setMinimumWidth(100)
        disable_wheel_event(ai_hunter_workers_spinbox)
        workers_layout.addWidget(ai_hunter_workers_spinbox)

        # CPU count display
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        cpu_hint = QLabel(f"(0 = use all {cpu_count} cores)")
        cpu_hint.setFont(QFont('Arial', 9))
        cpu_hint.setStyleSheet("color: gray;")
        workers_layout.addWidget(cpu_hint)
        workers_layout.addStretch()
        ai_hunter_layout.addWidget(workers_widget)

        # Quick preset buttons
        preset_widget = QWidget()
        preset_layout = QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)

        preset_label = QLabel("Quick presets:")
        preset_label.setFont(QFont('Arial', 9))
        preset_layout.addWidget(preset_label)
        preset_layout.addSpacing(10)

        all_cores_btn = QPushButton(f"All cores ({cpu_count})")
        all_cores_btn.setFont(QFont('Arial', 9))
        all_cores_btn.clicked.connect(lambda: ai_hunter_workers_spinbox.setValue(0))
        preset_layout.addWidget(all_cores_btn)

        half_cores_btn = QPushButton("Half cores")
        half_cores_btn.setFont(QFont('Arial', 9))
        half_cores_btn.clicked.connect(lambda: ai_hunter_workers_spinbox.setValue(max(1, cpu_count // 2)))
        preset_layout.addWidget(half_cores_btn)

        four_cores_btn = QPushButton("4 cores")
        four_cores_btn.setFont(QFont('Arial', 9))
        four_cores_btn.clicked.connect(lambda: ai_hunter_workers_spinbox.setValue(4))
        preset_layout.addWidget(four_cores_btn)

        eight_cores_btn = QPushButton("8 cores")
        eight_cores_btn.setFont(QFont('Arial', 9))
        eight_cores_btn.clicked.connect(lambda: ai_hunter_workers_spinbox.setValue(8))
        preset_layout.addWidget(eight_cores_btn)

        single_thread_btn = QPushButton("Single thread")
        single_thread_btn.setFont(QFont('Arial', 9))
        single_thread_btn.clicked.connect(lambda: ai_hunter_workers_spinbox.setValue(1))
        preset_layout.addWidget(single_thread_btn)

        preset_layout.addStretch()
        ai_hunter_layout.addWidget(preset_widget)

        # Performance tips
        tips_text = "Performance Tips:\n" + \
                    f"• Your system has {cpu_count} CPU cores available\n" + \
                    "• Using all cores provides maximum speed but may slow other applications\n" + \
                    "• 4-8 cores usually provides good balance of speed and system responsiveness\n" + \
                    "• Single thread (1) disables parallel processing for debugging"

        tips_label = QLabel(tips_text)
        tips_label.setFont(QFont('Arial', 9))
        tips_label.setStyleSheet("color: gray;")
        tips_label.setWordWrap(True)
        tips_label.setMaximumWidth(700)
        tips_label.setContentsMargins(20, 10, 0, 0)
        ai_hunter_layout.addWidget(tips_label)


        
        def save_settings():
            """Save QA scanner settings with comprehensive debugging"""
            try:
                # Check if debug mode is enabled
                debug_mode = self.config.get('show_debug_buttons', False)
                
                if debug_mode:
                    self.append_log("🔍 [DEBUG] Starting QA Scanner settings save process...")
                
                # Helper to get the selected radio button value
                def get_selected_radio_value(radio_button_list):
                    for rb, value in radio_button_list:
                        if rb.isChecked():
                            return value
                    return None
                
                # Core QA Settings with debugging
                core_settings_to_save = {
                    'foreign_char_threshold': (threshold_spinbox, lambda x: x.value()),
                    'excluded_characters': (excluded_text, lambda x: x.toPlainText().strip()),
                    'target_language': (target_language_combo, lambda x: x.currentText().lower()),
                    'check_encoding_issues': (check_encoding_checkbox, lambda x: x.isChecked()),
                    'check_repetition': (check_repetition_checkbox, lambda x: x.isChecked()),
                    'check_translation_artifacts': (check_artifacts_checkbox, lambda x: x.isChecked()),
                    'check_glossary_leakage': (check_glossary_checkbox, lambda x: x.isChecked()),
                    'min_file_length': (min_length_spinbox, lambda x: x.value()),
                    'report_format': (format_radio_buttons, get_selected_radio_value),
                    'auto_save_report': (auto_save_checkbox, lambda x: x.isChecked()),
                    'check_word_count_ratio': (check_word_count_checkbox, lambda x: x.isChecked()),
                    'check_multiple_headers': (check_multiple_headers_checkbox, lambda x: x.isChecked()),
                    'warn_name_mismatch': (warn_mismatch_checkbox, lambda x: x.isChecked()),
                    'check_missing_html_tag': (check_missing_html_tag_checkbox, lambda x: x.isChecked()),
                    'check_paragraph_structure': (check_paragraph_structure_checkbox, lambda x: x.isChecked()),
                    'check_invalid_nesting': (check_invalid_nesting_checkbox, lambda x: x.isChecked()),
                }
                
                failed_core_settings = []
                for setting_name, (var_obj, converter) in core_settings_to_save.items():
                    try:
                        old_value = qa_settings.get(setting_name, '<NOT SET>')
                        new_value = converter(var_obj)
                        qa_settings[setting_name] = new_value
                        
                        if debug_mode:
                            if old_value != new_value:
                                self.append_log(f"🔍 [DEBUG] QA {setting_name}: '{old_value}' → '{new_value}'")
                            else:
                                self.append_log(f"🔍 [DEBUG] QA {setting_name}: unchanged ('{new_value}')")
                            
                    except Exception as e:
                        failed_core_settings.append(f"{setting_name} ({str(e)})")
                        if debug_mode:
                            self.append_log(f"❌ [DEBUG] Failed to save QA {setting_name}: {e}")
                
                if failed_core_settings and debug_mode:
                    self.append_log(f"⚠️ [DEBUG] Failed QA core settings: {', '.join(failed_core_settings)}")
                
                # Cache settings with debugging
                if debug_mode:
                    self.append_log("🔍 [DEBUG] Saving QA cache settings...")
                cache_settings_to_save = {
                    'cache_enabled': (cache_enabled_checkbox, lambda x: x.isChecked()),
                    'cache_auto_size': (auto_size_checkbox, lambda x: x.isChecked()),
                    'cache_show_stats': (show_stats_checkbox, lambda x: x.isChecked()),
                }
                
                failed_cache_settings = []
                for setting_name, (var_obj, converter) in cache_settings_to_save.items():
                    try:
                        old_value = qa_settings.get(setting_name, '<NOT SET>')
                        new_value = converter(var_obj)
                        qa_settings[setting_name] = new_value
                        
                        if debug_mode:
                            if old_value != new_value:
                                self.append_log(f"🔍 [DEBUG] QA {setting_name}: '{old_value}' → '{new_value}'")
                            else:
                                self.append_log(f"🔍 [DEBUG] QA {setting_name}: unchanged ('{new_value}')")
                    except Exception as e:
                        failed_cache_settings.append(f"{setting_name} ({str(e)})")
                        if debug_mode:
                            self.append_log(f"❌ [DEBUG] Failed to save QA {setting_name}: {e}")
                
                # Save individual cache sizes with debugging
                saved_cache_vars = []
                failed_cache_vars = []
                for cache_name, cache_spinbox in cache_spinboxes.items():
                    try:
                        cache_key = f'cache_{cache_name}'
                        old_value = qa_settings.get(cache_key, '<NOT SET>')
                        new_value = cache_spinbox.value()
                        qa_settings[cache_key] = new_value
                        saved_cache_vars.append(cache_name)
                        
                        if debug_mode and old_value != new_value:
                            self.append_log(f"🔍 [DEBUG] QA {cache_key}: '{old_value}' → '{new_value}'")
                    except Exception as e:
                        failed_cache_vars.append(f"{cache_name} ({str(e)})")
                        if debug_mode:
                            self.append_log(f"❌ [DEBUG] Failed to save QA cache_{cache_name}: {e}")
                
                if debug_mode:
                    if saved_cache_vars:
                        self.append_log(f"🔍 [DEBUG] Saved {len(saved_cache_vars)} cache settings: {', '.join(saved_cache_vars)}")
                    if failed_cache_vars:
                        self.append_log(f"⚠️ [DEBUG] Failed cache settings: {', '.join(failed_cache_vars)}")
                
                # AI Hunter config with debugging
                if debug_mode:
                    self.append_log("🔍 [DEBUG] Saving AI Hunter config...")
                try:
                    if 'ai_hunter_config' not in self.config:
                        self.config['ai_hunter_config'] = {}
                        if debug_mode:
                            self.append_log("🔍 [DEBUG] Created new ai_hunter_config section")
                    
                    old_workers = self.config['ai_hunter_config'].get('ai_hunter_max_workers', '<NOT SET>')
                    new_workers = ai_hunter_workers_spinbox.value()
                    self.config['ai_hunter_config']['ai_hunter_max_workers'] = new_workers
                    
                    if debug_mode:
                        if old_workers != new_workers:
                            self.append_log(f"🔍 [DEBUG] AI Hunter max_workers: '{old_workers}' → '{new_workers}'")
                        else:
                            self.append_log(f"🔍 [DEBUG] AI Hunter max_workers: unchanged ('{new_workers}')")
                        
                except Exception as e:
                    if debug_mode:
                        self.append_log(f"❌ [DEBUG] Failed to save AI Hunter config: {e}")
    
                # Validate and save paragraph threshold with debugging
                if debug_mode:
                    self.append_log("🔍 [DEBUG] Validating paragraph threshold...")
                try:
                    threshold_value = paragraph_threshold_spinbox.value()
                    old_threshold = qa_settings.get('paragraph_threshold', '<NOT SET>')
                    
                    if 0 <= threshold_value <= 100:
                        new_threshold = threshold_value / 100.0  # Convert to decimal
                        qa_settings['paragraph_threshold'] = new_threshold
                        
                        if debug_mode:
                            if old_threshold != new_threshold:
                                self.append_log(f"🔍 [DEBUG] QA paragraph_threshold: '{old_threshold}' → '{new_threshold}' ({threshold_value}%)")
                            else:
                                self.append_log(f"🔍 [DEBUG] QA paragraph_threshold: unchanged ('{new_threshold}' / {threshold_value}%)")
                    else:
                        raise ValueError("Threshold must be between 0 and 100")
                        
                except (ValueError, Exception) as e:
                    # Default to 30% if invalid
                    qa_settings['paragraph_threshold'] = 0.3
                    if debug_mode:
                        self.append_log(f"❌ [DEBUG] Invalid paragraph threshold ({e}), using default 30%")
                    self.append_log("⚠️ Invalid paragraph threshold, using default 30%")

                # Save to main config with debugging
                if debug_mode:
                    self.append_log("🔍 [DEBUG] Saving QA settings to main config...")
                try:
                    old_qa_config = self.config.get('qa_scanner_settings', {})
                    self.config['qa_scanner_settings'] = qa_settings
                    
                    if debug_mode:
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
                    if debug_mode:
                        self.append_log(f"❌ [DEBUG] Failed to update main config: {e}")
                
                # Environment variables setup for QA Scanner
                if debug_mode:
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
                            
                            if debug_mode:
                                if old_value != new_value:
                                    self.append_log(f"🔍 [DEBUG] ENV {env_key}: '{old_value}' → '{new_value}'")
                                else:
                                    self.append_log(f"🔍 [DEBUG] ENV {env_key}: unchanged ('{new_value}')")
                                
                        except Exception as e:
                            if debug_mode:
                                self.append_log(f"❌ [DEBUG] Failed to set {env_key}: {e}")
                    
                    if debug_mode:
                        self.append_log(f"🔍 [DEBUG] Successfully set {len(qa_env_vars_set)} QA environment variables")
                    
                except Exception as e:
                    if debug_mode:
                        self.append_log(f"❌ [DEBUG] QA environment variable setup failed: {e}")
                        import traceback
                        self.append_log(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
                
                # Call save_config with show_message=False to avoid the error
                if debug_mode:
                    self.append_log("🔍 [DEBUG] Calling main save_config method...")
                try:
                    self.save_config(show_message=False)
                    if debug_mode:
                        self.append_log("🔍 [DEBUG] Main save_config completed successfully")
                except Exception as e:
                    if debug_mode:
                        self.append_log(f"❌ [DEBUG] Main save_config failed: {e}")
                    raise
                
                # Final QA environment variable verification
                if debug_mode:
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
                dialog.accept()
                
            except Exception as e:
                # Get debug_mode again in case of early exception
                debug_mode = self.config.get('show_debug_buttons', False)
                if debug_mode:
                    self.append_log(f"❌ [DEBUG] QA save_settings full exception: {str(e)}")
                    import traceback
                    self.append_log(f"❌ [DEBUG] QA save_settings traceback: {traceback.format_exc()}")
                self.append_log(f"❌ Error saving QA settings: {str(e)}")
                QMessageBox.critical(dialog, "Error", f"Failed to save settings: {str(e)}")
        
        def reset_defaults():
            """Reset to default settings"""
            result = QMessageBox.question(
                dialog,
                "Reset to Defaults",
                "Are you sure you want to reset all settings to defaults?",
                QMessageBox.Yes | QMessageBox.No
            )
            if result == QMessageBox.Yes:
                threshold_spinbox.setValue(10)
                excluded_text.clear()
                target_language_combo.setCurrentText('English')
                check_encoding_checkbox.setChecked(False)
                check_repetition_checkbox.setChecked(True)
                check_artifacts_checkbox.setChecked(False)

                check_glossary_checkbox.setChecked(True)
                min_length_spinbox.setValue(0)
                # Set 'detailed' radio button as checked
                for rb, value in format_radio_buttons:
                    rb.setChecked(value == 'detailed')
                auto_save_checkbox.setChecked(True)
                check_word_count_checkbox.setChecked(False)
                check_multiple_headers_checkbox.setChecked(True)
                warn_mismatch_checkbox.setChecked(False)
                check_missing_html_tag_checkbox.setChecked(True)
                check_paragraph_structure_checkbox.setChecked(True)
                check_invalid_nesting_checkbox.setChecked(False)
                paragraph_threshold_spinbox.setValue(30)  # 30% default
                
                # Reset cache settings
                cache_enabled_checkbox.setChecked(True)
                auto_size_checkbox.setChecked(False)
                show_stats_checkbox.setChecked(False)
                
                # Reset cache sizes to defaults
                for cache_name, default_value in cache_defaults.items():
                    cache_spinboxes[cache_name].setValue(default_value)
                    
                ai_hunter_workers_spinbox.setValue(1)
        
        scroll_layout.addStretch()
        
        # Create fixed bottom button section (outside scroll area)
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(20, 15, 20, 15)
        
        save_btn = QPushButton("Save Settings")
        save_btn.setMinimumWidth(120)
        save_btn.setStyleSheet("background-color: #28a745; color: white; padding: 8px; font-weight: bold;")
        save_btn.clicked.connect(save_settings)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(120)
        cancel_btn.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
        cancel_btn.clicked.connect(lambda: [dialog._cleanup_scrolling(), dialog.reject()])
        button_layout.addWidget(cancel_btn)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.setMinimumWidth(120)
        reset_btn.setStyleSheet("background-color: #ffc107; color: black; padding: 8px;")
        reset_btn.clicked.connect(reset_defaults)
        button_layout.addWidget(reset_btn)
        
        # Add button widget to main layout (not scroll layout)
        main_layout.addWidget(button_widget)
        
        # Show the dialog (PySide6 handles sizing automatically)
        # Note: The dialog size is already set in the constructor (800x600)
        
        # Add a dummy _cleanup_scrolling method for compatibility
        dialog._cleanup_scrolling = lambda: None
        
        # Handle window close - just cleanup, don't call reject() to avoid recursion
        def handle_close():
            dialog._cleanup_scrolling()
        dialog.rejected.connect(handle_close)
        
        # Show the dialog with fade animation and return result
        try:
            from dialog_animations import exec_dialog_with_fade
            return exec_dialog_with_fade(dialog, duration=250)
        except Exception:
            return dialog.exec()


def show_custom_detection_dialog(parent=None):
    """
    Standalone function to show the custom detection settings dialog.
    Returns a dictionary with the settings if user confirms, None if cancelled.
    
    This function can be called from anywhere, including scan_html_folder.py
    """
    from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QLabel, QPushButton, 
                                   QVBoxLayout, QHBoxLayout, QScrollArea, QGroupBox,
                                   QCheckBox, QSpinBox, QSlider, QMessageBox, QSizePolicy)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QIcon
    import os
    
    # Create dialog
    custom_dialog = QDialog(parent)
    custom_dialog.setWindowTitle("Custom Mode Settings")
    custom_dialog.setModal(True)
    
    # Set dialog size
    screen = QApplication.primaryScreen().geometry()
    custom_width = int(screen.width() * 0.41)
    custom_height = int(screen.height() * 0.60)
    custom_dialog.resize(custom_width, custom_height)
    
    # Set window icon
    try:
        # Try to find the icon in common locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico'),
            os.path.join(os.getcwd(), 'Halgakos.ico'),
        ]
        for ico_path in possible_paths:
            if os.path.isfile(ico_path):
                custom_dialog.setWindowIcon(QIcon(ico_path))
                break
    except Exception:
        pass
    
    # Main layout
    dialog_layout = QVBoxLayout(custom_dialog)
    
    # Scroll area
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    # Scrollable content widget
    scroll_widget = QWidget()
    scroll_layout = QVBoxLayout(scroll_widget)
    scroll.setWidget(scroll_widget)
    dialog_layout.addWidget(scroll)
    
    # Default settings
    custom_settings = {
        'text_similarity': 85,
        'semantic_analysis': 80,
        'structural_patterns': 90,
        'word_overlap': 75,
        'minhash_similarity': 80,
        'consecutive_chapters': 2,
        'check_all_pairs': False,
        'sample_size': 3000,
        'min_text_length': 500
    }
    
    # Store widget references
    custom_widgets = {}
    
    # Title
    title_label = QLabel("Configure Custom Detection Settings")
    title_label.setFont(QFont('Arial', 20, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    scroll_layout.addWidget(title_label)
    scroll_layout.addSpacing(20)
    
    # Detection Thresholds Section
    threshold_group = QGroupBox("Detection Thresholds (%)")
    threshold_group.setFont(QFont('Arial', 12, QFont.Bold))
    threshold_layout = QVBoxLayout(threshold_group)
    threshold_layout.setContentsMargins(25, 25, 25, 25)
    scroll_layout.addWidget(threshold_group)
    
    threshold_descriptions = {
        'text_similarity': ('Text Similarity', 'Character-by-character comparison'),
        'semantic_analysis': ('Semantic Analysis', 'Meaning and context matching'),
        'structural_patterns': ('Structural Patterns', 'Document structure similarity'),
        'word_overlap': ('Word Overlap', 'Common words between texts'),
        'minhash_similarity': ('MinHash Similarity', 'Fast approximate matching')
    }
    
    # Create percentage labels dictionary
    percentage_labels = {}
    
    for setting_key, (label_text, description) in threshold_descriptions.items():
        # Container for each threshold
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 8, 0, 8)
        
        # Left side - labels
        label_widget = QWidget()
        label_layout = QVBoxLayout(label_widget)
        label_layout.setContentsMargins(0, 0, 0, 0)
        
        main_label = QLabel(f"{label_text} - {description}:")
        main_label.setFont(QFont('Arial', 11))
        label_layout.addWidget(main_label)
        label_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row_layout.addWidget(label_widget)
        
        # Right side - slider and percentage
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(20, 0, 0, 0)
        
        # Create slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(100)
        slider.setValue(custom_settings[setting_key])
        slider.setMinimumWidth(300)
        slider.wheelEvent = lambda event: event.ignore()
        slider_layout.addWidget(slider)
        
        # Percentage label
        percentage_label = QLabel(f"{custom_settings[setting_key]}%")
        percentage_label.setFont(QFont('Arial', 12, QFont.Bold))
        percentage_label.setMinimumWidth(50)
        percentage_label.setAlignment(Qt.AlignRight)
        slider_layout.addWidget(percentage_label)
        percentage_labels[setting_key] = percentage_label
        
        row_layout.addWidget(slider_widget)
        threshold_layout.addWidget(row_widget)
        
        # Store slider widget reference
        custom_widgets[setting_key] = slider
        
        # Update percentage label when slider moves
        def create_update_function(key, label, settings_dict):
            def update_percentage(value):
                settings_dict[key] = value
                label.setText(f"{value}%")
            return update_percentage
        
        update_func = create_update_function(setting_key, percentage_label, custom_settings)
        slider.valueChanged.connect(update_func)
    
    scroll_layout.addSpacing(15)
    
    # Processing Options Section
    options_group = QGroupBox("Processing Options")
    options_group.setFont(QFont('Arial', 12, QFont.Bold))
    options_layout = QVBoxLayout(options_group)
    options_layout.setContentsMargins(20, 20, 20, 20)
    scroll_layout.addWidget(options_group)
    
    # Consecutive chapters option
    consec_widget = QWidget()
    consec_layout = QHBoxLayout(consec_widget)
    consec_layout.setContentsMargins(0, 5, 0, 5)
    
    consec_label = QLabel("Consecutive chapters to check:")
    consec_label.setFont(QFont('Arial', 11))
    consec_layout.addWidget(consec_label)
    
    consec_spinbox = QSpinBox()
    consec_spinbox.setMinimum(1)
    consec_spinbox.setMaximum(10)
    consec_spinbox.setValue(custom_settings['consecutive_chapters'])
    consec_spinbox.setMinimumWidth(100)
    consec_spinbox.wheelEvent = lambda event: event.ignore()
    consec_layout.addWidget(consec_spinbox)
    consec_layout.addStretch()
    options_layout.addWidget(consec_widget)
    custom_widgets['consecutive_chapters'] = consec_spinbox
    
    # Sample size option
    sample_widget = QWidget()
    sample_layout = QHBoxLayout(sample_widget)
    sample_layout.setContentsMargins(0, 5, 0, 5)
    
    sample_label = QLabel("Sample size for comparison (characters):")
    sample_label.setFont(QFont('Arial', 11))
    sample_layout.addWidget(sample_label)
    
    sample_spinbox = QSpinBox()
    sample_spinbox.setMinimum(1000)
    sample_spinbox.setMaximum(10000)
    sample_spinbox.setSingleStep(500)
    sample_spinbox.setValue(custom_settings['sample_size'])
    sample_spinbox.setMinimumWidth(100)
    sample_spinbox.wheelEvent = lambda event: event.ignore()
    sample_layout.addWidget(sample_spinbox)
    sample_layout.addStretch()
    options_layout.addWidget(sample_widget)
    custom_widgets['sample_size'] = sample_spinbox
    
    # Minimum text length option
    min_length_widget = QWidget()
    min_length_layout = QHBoxLayout(min_length_widget)
    min_length_layout.setContentsMargins(0, 5, 0, 5)
    
    min_length_label = QLabel("Minimum text length to process (characters):")
    min_length_label.setFont(QFont('Arial', 11))
    min_length_layout.addWidget(min_length_label)
    
    min_length_spinbox = QSpinBox()
    min_length_spinbox.setMinimum(100)
    min_length_spinbox.setMaximum(5000)
    min_length_spinbox.setSingleStep(100)
    min_length_spinbox.setValue(custom_settings['min_text_length'])
    min_length_spinbox.setMinimumWidth(100)
    min_length_spinbox.wheelEvent = lambda event: event.ignore()
    min_length_layout.addWidget(min_length_spinbox)
    min_length_layout.addStretch()
    options_layout.addWidget(min_length_widget)
    custom_widgets['min_text_length'] = min_length_spinbox
    
    # Check all file pairs option
    check_all_checkbox = QCheckBox("Check all file pairs (slower but more thorough)")
    check_all_checkbox.setChecked(custom_settings['check_all_pairs'])
    options_layout.addWidget(check_all_checkbox)
    custom_widgets['check_all_pairs'] = check_all_checkbox
    
    scroll_layout.addSpacing(30)
    
    # Button layout
    button_widget = QWidget()
    button_layout = QHBoxLayout(button_widget)
    button_layout.addStretch()
    scroll_layout.addWidget(button_widget)
    
    # Flag to track if settings were confirmed
    settings_confirmed = False
    result_settings = None
    
    def confirm_settings():
        """Confirm settings and close dialog"""
        nonlocal settings_confirmed, result_settings
        
        result_settings = {
            'text_similarity': custom_widgets['text_similarity'].value(),
            'semantic_analysis': custom_widgets['semantic_analysis'].value(),
            'structural_patterns': custom_widgets['structural_patterns'].value(),
            'word_overlap': custom_widgets['word_overlap'].value(),
            'minhash_similarity': custom_widgets['minhash_similarity'].value(),
            'consecutive_chapters': custom_widgets['consecutive_chapters'].value(),
            'check_all_pairs': custom_widgets['check_all_pairs'].isChecked(),
            'sample_size': custom_widgets['sample_size'].value(),
            'min_text_length': custom_widgets['min_text_length'].value()
        }
        settings_confirmed = True
        custom_dialog.accept()
    
    def reset_to_defaults():
        """Reset all values to defaults"""
        reply = QMessageBox.question(custom_dialog, "Reset to Defaults", 
                                   "Reset all values to default settings?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            custom_widgets['text_similarity'].setValue(85)
            custom_widgets['semantic_analysis'].setValue(80)
            custom_widgets['structural_patterns'].setValue(90)
            custom_widgets['word_overlap'].setValue(75)
            custom_widgets['minhash_similarity'].setValue(80)
            custom_widgets['consecutive_chapters'].setValue(2)
            custom_widgets['check_all_pairs'].setChecked(False)
            custom_widgets['sample_size'].setValue(3000)
            custom_widgets['min_text_length'].setValue(500)
    
    # Create buttons
    cancel_btn = QPushButton("Cancel")
    cancel_btn.setMinimumWidth(120)
    cancel_btn.clicked.connect(custom_dialog.reject)
    button_layout.addWidget(cancel_btn)
    
    reset_btn = QPushButton("Reset Defaults")
    reset_btn.setMinimumWidth(120)
    reset_btn.clicked.connect(reset_to_defaults)
    button_layout.addWidget(reset_btn)
    
    start_btn = QPushButton("Start Scan")
    start_btn.setMinimumWidth(120)
    start_btn.setStyleSheet("""
        QPushButton {
            background-color: #28a745;
            color: white;
            border: 1px solid #28a745;
            padding: 6px 12px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
    """)
    start_btn.clicked.connect(confirm_settings)
    button_layout.addWidget(start_btn)
    
    button_layout.addStretch()
    
    # Show dialog with fade animation and return result
    try:
        from dialog_animations import exec_dialog_with_fade
        exec_dialog_with_fade(custom_dialog, duration=250)
    except Exception:
        custom_dialog.exec()
    
    # Return settings if confirmed, None otherwise
    return result_settings if settings_confirmed else None
