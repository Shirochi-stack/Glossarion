# async_api_processor.py
"""
Asynchronous API Processing for Glossarion
Implements batch API processing with 50% discount from supported providers.
This is SEPARATE from the existing batch processing (parallel API calls).

Supported Providers with Async/Batch APIs (50% discount):
- Gemini (Batch API)
- Anthropic (Message Batches API)
- OpenAI (Batch API)
- Mistral (Batch API)
- Amazon Bedrock (Batch Inference)
- Groq (Batch API)

Providers without Async APIs:
- DeepSeek (no batch API)
- Cohere (only batch embeddings, not completions)

U7: the GUI-free core (AsyncAPIProcessor, the job model and the dialog's workflow methods)
lives in async_batch_core; this module re-exports those names and keeps the Qt dialog as a
thin view (AsyncProcessingDialog = AsyncBatchJobMixin + widgets + the Qt hook statements).
It imports without PySide6 (the dialog then needs PySide6 at call time).
"""

import os
import sys
import re
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import json
import time
import threading
import logging
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
try:
    from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                                    QLabel, QPushButton, QCheckBox, QSpinBox, QTreeWidget, QTreeWidgetItem,
                                    QScrollArea, QProgressBar, QGroupBox, QFrame, QMessageBox, QMenu, QApplication,
                                    QAbstractItemView)
    from PySide6.QtCore import Qt, QTimer, Signal, QObject, QEventLoop
    from PySide6.QtGui import QIcon, QBrush, QColor
except ImportError:  # GUI-free import (Glossarion Mobile, scripts): the core names stay importable
    QDialog = QWidget = QVBoxLayout = QHBoxLayout = QGridLayout = None
    QLabel = QPushButton = QCheckBox = QSpinBox = QTreeWidget = QTreeWidgetItem = None
    QScrollArea = QProgressBar = QGroupBox = QFrame = QMessageBox = QMenu = QApplication = None
    QAbstractItemView = None
    Qt = QTimer = Signal = QObject = QEventLoop = None
    QIcon = QBrush = QColor = None
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import uuid
from pathlib import Path
from html_output_utils import ensure_utf8_html_document
from epub_package import find_epub_opf_member

# U7: moved to async_batch_core (GUI-free); re-exported under the old names.
from async_batch_core import (
    AsyncAPIProcessor,
    AsyncAPIStatus,
    AsyncBatchJobMixin,
    AsyncJobInfo,
    HAS_ANTHROPIC,
    HAS_GEMINI,
    HAS_OPENAI,
    TextFileProcessor,
    _clamp_antigravity_output_tokens,
    _clamp_output_tokens_for_selected_model,
    _is_antigravity_model_name,
    async_support_status,
    gui_model_name,
    job_display_row,
    refresh_pending_job_statuses,
    selected_job_progress,
    tiktoken,
)
if HAS_GEMINI:
    from async_batch_core import genai
if HAS_ANTHROPIC:
    from async_batch_core import anthropic
if HAS_OPENAI:
    from async_batch_core import openai

logger = logging.getLogger(__name__)


class AsyncProcessingDialog(AsyncBatchJobMixin):
    """GUI dialog for async processing (the workflow methods live in AsyncBatchJobMixin)"""

    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay - from manga integration"""
        from PySide6.QtWidgets import QCheckBox, QLabel
        from PySide6.QtCore import Qt, QTimer
        
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet("""
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
        
        # Create checkmark overlay
        checkmark = QLabel("✓", checkbox)
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
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)  # Make checkmark click-through
        
        # Position checkmark properly after widget is shown
        def position_checkmark():
            try:
                # Check if checkmark still exists and is valid
                if checkmark and not checkmark.isHidden() or True:  # Always try to set geometry
                    checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                # Widget was already deleted
                pass
        
        # Show/hide checkmark based on checked state
        def update_checkmark():
            try:
                # Check if both widgets still exist
                if checkbox and checkmark:
                    if checkbox.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                # Widget was already deleted
                pass
        
        checkbox.stateChanged.connect(update_checkmark)
        # Delay initial positioning to ensure widget is properly rendered
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark()))
        
        return checkbox
    
    def __init__(self, parent, translator_gui):
        """Initialize dialog
        
        Args:
            parent: Parent window
            translator_gui: Reference to main TranslatorGUI instance
        """
        self.parent = parent
        self.gui = translator_gui
        
        # Fix for PyInstaller - ensure processor uses correct directory
        self.processor = AsyncAPIProcessor(translator_gui)
        
        # If running as exe, update the jobs file path
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            application_path = os.path.dirname(sys.executable)
            self.processor.jobs_file = os.path.join(application_path, 'async_jobs.json')
            # Reload jobs from the correct location
            self.processor._load_jobs()
        
        self.selected_job_id = None
        self.polling_jobs = set()  # Track which jobs are being polled
        
        self._create_dialog()
        self._refresh_jobs_list()
        
    def _create_dialog(self):
        """Create the async processing dialog"""
        # Create main dialog
        self.dialog = QDialog(self.parent)
        self.dialog.setWindowTitle("Async Batch Processing (50% Discount)")
        self.dialog.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
        """)

        # Override close behavior: hide instead of closing to preserve state
        def _on_close(event):
            try:
                event.ignore()
                self.dialog.hide()
                return
            except Exception:
                pass
            event.accept()
        self.dialog.closeEvent = _on_close
        
        # Set icon if available
        try:
            icon_path = os.path.join(self.gui.base_dir, 'Halgakos.ico')
            if os.path.exists(icon_path):
                self.dialog.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
        
        # Main layout
        main_layout = QVBoxLayout(self.dialog)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create scrollable content widget
        scrollable_widget = QWidget()
        content_layout = QVBoxLayout(scrollable_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        
        # Top section - Information and controls
        self._create_info_section(scrollable_widget)
        
        # Middle section - Configuration
        self._create_config_section(scrollable_widget)
        
        # Bottom section - Active jobs
        self._create_jobs_section(scrollable_widget)
        
        # Set scroll area widget
        scroll_area.setWidget(scrollable_widget)
        main_layout.addWidget(scroll_area)
        
        # Button frame at bottom of dialog
        self._create_button_frame(self.dialog)
        
        # Load active jobs
        self._refresh_jobs_list()
        
        # Size and position dialog - give wider default
        app = QApplication.instance()
        if app:
            screen = app.primaryScreen().availableGeometry()
            dialog_width = int(screen.width() * 0.6)
            dialog_height = int(screen.height() * 0.8)
            self.dialog.resize(dialog_width, dialog_height)
            # Ensure it doesn't shrink too small for new columns
            self.dialog.setMinimumWidth(max(900, int(screen.width() * 0.6)))
            
            # Center the dialog
            dialog_x = screen.x() + (screen.width() - dialog_width) // 2
            dialog_y = screen.y() + (screen.height() - dialog_height) // 2
            self.dialog.move(dialog_x, dialog_y)
       
        # Start auto refresh
        self._start_auto_refresh(30)
        
        # Show dialog
        self.dialog.show()
        
    def _create_info_section(self, parent):
        """Create information section"""
        info_group = QGroupBox("Async Processing Information")
        info_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                border: 0.1em solid #555555;
                border-radius: 0.25em;
                margin-top: 0.5em;
                padding: 0.75em;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0.5em;
                padding: 0 0.25em;
                color: #ffffff;
            }
        """)
        info_layout = QVBoxLayout()
        info_group.setLayout(info_layout)
        
        # Model and provider info
        model_layout = QHBoxLayout()
        
        model_label_text = QLabel("Current Model:")
        model_label_text.setStyleSheet("font-size: 10pt; color: #ffffff;")
        model_layout.addWidget(model_label_text)
        
        # Get model name from GUI - handle both tkinter and PySide6
        model_name = gui_model_name(self.gui)
        self.model_label = QLabel(model_name)
        self.model_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #ffffff;")
        model_layout.addWidget(self.model_label)
        model_layout.addSpacing(20)
        
        # Check if model supports async
        supported, status_text = async_support_status(self.processor, model_name)
        if supported:
            self.status_label = QLabel(status_text)
            self.status_label.setStyleSheet("color: #28a745; font-size: 10pt; font-weight: bold;")
        else:
            self.status_label = QLabel(status_text)
            self.status_label.setStyleSheet("color: #dc3545; font-size: 10pt; font-weight: bold;")
            
        model_layout.addWidget(self.status_label)
        model_layout.addStretch()
        info_layout.addLayout(model_layout)
        
        # Cost estimation
        cost_label = QLabel("Cost Estimation:")
        cost_label.setStyleSheet("font-size: 11pt; font-weight: bold; margin-top: 10px; color: #ffffff;")
        info_layout.addWidget(cost_label)
        
        self.cost_info_label = QLabel("Select chapters to see cost estimate")
        self.cost_info_label.setStyleSheet("font-size: 10pt; color: #aaaaaa; padding: 0.25em;")
        self.cost_info_label.setWordWrap(True)
        info_layout.addWidget(self.cost_info_label)
        
        # Add to parent layout
        parent.layout().addWidget(info_group)
        
    def _refresh_model_info(self):
        """Refresh model name and support status from current GUI selection"""
        try:
            model_name = gui_model_name(self.gui)

            self.model_label.setText(model_name)

            supported, status_text = async_support_status(self.processor, model_name)
            if supported:
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("color: #28a745; font-size: 10pt; font-weight: bold;")
            else:
                self.status_label.setText(status_text)
                self.status_label.setStyleSheet("color: #dc3545; font-size: 10pt; font-weight: bold;")
        except Exception as e:
            print(f"[ASYNC] Failed to refresh model info: {e}")
    
    def _create_config_section(self, parent):
        """Create configuration section"""
        config_group = QGroupBox("Async Processing Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                border: 0.1em solid #555555;
                border-radius: 0.25em;
                margin-top: 0.5em;
                padding: 0.75em;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0.5em;
                padding: 0 0.25em;
                color: #ffffff;
            }
        """)
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)
        
        # Wait for completion checkbox using styled version
        self.wait_for_completion_checkbox = self._create_styled_checkbox("Wait for completion (blocks GUI)")
        # Load from config
        wait_value = self.gui.config.get('async_wait_for_completion', False)
        print(f"[ASYNC_DEBUG] Loading async_wait_for_completion: {wait_value}")
        self.wait_for_completion_checkbox.setChecked(wait_value)
        # Save to config when changed
        def _on_wait_changed(checked):
            self.gui.config['async_wait_for_completion'] = checked
            self.gui.async_wait_for_completion_var = checked
            print(f"[ASYNC_DEBUG] Saving async_wait_for_completion: {checked}")
            self.gui.save_config(show_message=False)
        self.wait_for_completion_checkbox.toggled.connect(_on_wait_changed)
        config_layout.addWidget(self.wait_for_completion_checkbox)
        
        # Poll interval
        poll_layout = QHBoxLayout()
        poll_label = QLabel("Poll interval (seconds):")
        poll_label.setStyleSheet("font-size: 10pt; color: #ffffff;")
        poll_layout.addWidget(poll_label)
        
        self.poll_interval_spinbox = QSpinBox()
        self.poll_interval_spinbox.setMinimum(10)
        self.poll_interval_spinbox.setMaximum(600)
        # Load from config
        poll_value = int(self.gui.config.get('async_poll_interval', 60))
        print(f"[ASYNC_DEBUG] Loading async_poll_interval: {poll_value}")
        self.poll_interval_spinbox.setValue(poll_value)
        # Save to config when changed
        def _on_poll_changed(value):
            self.gui.config['async_poll_interval'] = value
            self.gui.async_poll_interval_var = value
            print(f"[ASYNC_DEBUG] Saving async_poll_interval: {value}")
            self.gui.save_config(show_message=False)
        self.poll_interval_spinbox.valueChanged.connect(_on_poll_changed)
        self.poll_interval_spinbox.setFixedWidth(100)
        # Disable mousewheel scrolling
        self.poll_interval_spinbox.wheelEvent = lambda event: None
        # Don't set any custom stylesheet - let it use default arrows
        poll_layout.addWidget(self.poll_interval_spinbox)
        poll_layout.addStretch()
        
        config_layout.addLayout(poll_layout)
        
        # Chapter selection info
        self.chapter_info_label = QLabel("Note: Async processing will skip chapters that require chunking")
        self.chapter_info_label.setStyleSheet("color: #ffa500; font-size: 9pt; padding: 0.25em;")
        self.chapter_info_label.setWordWrap(True)
        config_layout.addWidget(self.chapter_info_label)
        
        # Add to parent layout
        parent.layout().addWidget(config_group)
        
    def _create_jobs_section(self, parent):
        """Create active jobs section"""
        jobs_group = QGroupBox("Active Async Jobs")
        jobs_group.setStyleSheet("""
            QGroupBox {
                font-size: 11pt;
                font-weight: bold;
                border: 0.1em solid #555555;
                border-radius: 0.25em;
                margin-top: 0.5em;
                padding: 0.75em;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0.5em;
                padding: 0 0.25em;
                color: #ffffff;
            }
        """)
        jobs_layout = QVBoxLayout()
        jobs_group.setLayout(jobs_layout)
        
        # Jobs tree widget
        self.jobs_tree = QTreeWidget()
        self.jobs_tree.setColumnCount(8)
        self.jobs_tree.setHeaderLabels(["Job ID", "Provider", "Model", "Status", "Progress", "Created", "Source File", "Cost"])
        self.jobs_tree.setStyleSheet("""
            QTreeWidget {
                font-size: 10pt;
                background-color: #2b2b2b;
                alternate-background-color: #333333;
                border: 0.05em solid #555555;
                border-radius: 0.15em;
                color: #ffffff;
            }
            QTreeWidget::item {
                padding: 0.25em;
                color: #ffffff;
            }
            QTreeWidget::item:hover {
                background-color: #3d3d3d;
            }
            QTreeWidget::item:selected {
                background-color: #0078d7;
                color: white;
            }
            QHeaderView::section {
                background-color: #1e1e1e;
                color: #ffffff;
                padding: 0.25em;
                border: 0.05em solid #555555;
                font-weight: bold;
            }
        """)
        self.jobs_tree.setAlternatingRowColors(True)
        # Allow selecting multiple jobs (Ctrl/Cmd-click, Shift-click, or Ctrl+A)
        self.jobs_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.jobs_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Set column widths
        self.jobs_tree.setColumnWidth(0, 200)  # Job ID
        self.jobs_tree.setColumnWidth(1, 100)  # Provider
        self.jobs_tree.setColumnWidth(2, 150)  # Model
        self.jobs_tree.setColumnWidth(3, 100)  # Status
        self.jobs_tree.setColumnWidth(4, 150)  # Progress
        self.jobs_tree.setColumnWidth(5, 150)  # Created
        self.jobs_tree.setColumnWidth(6, 220)  # Source File
        self.jobs_tree.setColumnWidth(7, 100)  # Cost
        
        jobs_layout.addWidget(self.jobs_tree)
        
        # Add a progress bar for the selected job
        progress_layout = QHBoxLayout()
        progress_text = QLabel("Selected Job Progress:")
        progress_text.setStyleSheet("font-size: 10pt; font-weight: bold; color: #ffffff;")
        progress_layout.addWidget(progress_text)
        
        self.job_progress_bar = QProgressBar()
        self.job_progress_bar.setMinimum(0)
        self.job_progress_bar.setMaximum(100)
        self.job_progress_bar.setValue(0)
        self.job_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 0.1em solid #555555;
                border-radius: 0.25em;
                text-align: center;
                font-size: 9pt;
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 0.15em;
            }
        """)
        progress_layout.addWidget(self.job_progress_bar)
        
        self.progress_label = QLabel("0%")
        self.progress_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #aaaaaa;")
        progress_layout.addWidget(self.progress_label)
        
        jobs_layout.addLayout(progress_layout)
        
        # Create context menu
        self.jobs_context_menu = QMenu(self.jobs_tree)
        self.jobs_context_menu.addAction("Check Status", self._check_selected_status)
        self.jobs_context_menu.addAction("Retrieve Results", self._retrieve_selected_results)
        self.jobs_context_menu.addSeparator()
        self.jobs_context_menu.addAction("Delete", self._delete_selected_job)
        
        # Set context menu policy
        self.jobs_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.jobs_tree.customContextMenuRequested.connect(self._show_context_menu)
        
        # Connect selection change
        self.jobs_tree.itemSelectionChanged.connect(self._on_job_select)
        
        # Job action buttons
        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)
        
        button_style = """
            QPushButton {
                background-color: #495057;
                color: white;
                font-size: 10pt;
                font-weight: bold;
                padding: 0.5em 1em;
                border-radius: 0.25em;
                border: none;
                min-width: 6em;
            }
            QPushButton:hover {
                background-color: #3d4349;
            }
            QPushButton:pressed {
                background-color: #2d3238;
            }
        """
        
        check_status_btn = QPushButton("Check Status")
        check_status_btn.clicked.connect(self._check_selected_status)
        check_status_btn.setStyleSheet(button_style)
        action_layout.addWidget(check_status_btn)
        
        retrieve_btn = QPushButton("Retrieve Results")
        retrieve_btn.clicked.connect(self._retrieve_selected_results)
        retrieve_btn.setStyleSheet(button_style.replace("#495057", "#1e7e34").replace("#3d4349", "#19692c").replace("#2d3238", "#145523"))
        action_layout.addWidget(retrieve_btn)
        
        cancel_btn = QPushButton("Cancel Job")
        cancel_btn.clicked.connect(self._cancel_selected_job)
        cancel_btn.setStyleSheet(button_style.replace("#495057", "#e0a800").replace("#3d4349", "#c69500").replace("#2d3238", "#b38600"))
        action_layout.addWidget(cancel_btn)
        
        action_layout.addSpacing(30)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self._delete_selected_job)
        delete_btn.setStyleSheet(button_style.replace("#495057", "#bd2130").replace("#3d4349", "#a71d2a").replace("#2d3238", "#8b1924"))
        action_layout.addWidget(delete_btn)
        
        clear_btn = QPushButton("Clear Completed")
        clear_btn.clicked.connect(self._clear_completed_jobs)
        clear_btn.setStyleSheet(button_style)
        action_layout.addWidget(clear_btn)
        
        action_layout.addStretch()
        jobs_layout.addLayout(action_layout)
        
        # Add to parent layout
        parent.layout().addWidget(jobs_group)
    
    def _create_button_frame(self, parent):
        """Create bottom button frame"""
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 5, 10, 10)
        
        # Start processing button
        self.start_button = QPushButton("Start Async Processing")
        self.start_button.clicked.connect(self._start_processing)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #1e7e34;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 0.7em 1.5em;
                border-radius: 0.25em;
                border: none;
                min-width: 10em;
            }
            QPushButton:hover {
                background-color: #19692c;
                border: 1px solid #28a745;
            }
            QPushButton:pressed {
                background-color: #145523;
            }
            QPushButton:disabled {
                background-color: #5a6268;
                color: #999999;
            }
        """)
        button_layout.addWidget(self.start_button)
        
        # Estimate only button
        estimate_button = QPushButton("Estimate Cost Only")
        estimate_button.clicked.connect(self._estimate_cost)
        estimate_button.setStyleSheet("""
            QPushButton {
                background-color: #0056b3;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 0.7em 1.5em;
                border-radius: 0.25em;
                border: none;
                min-width: 9em;
            }
            QPushButton:hover {
                background-color: #004a9f;
                border: 1px solid #007bff;
            }
            QPushButton:pressed {
                background-color: #003d82;
            }
        """)
        button_layout.addWidget(estimate_button)
        
        button_layout.addStretch()
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.dialog.close)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #495057;
                color: white;
                font-size: 10pt;
                font-weight: bold;
                padding: 0.7em 1.5em;
                border-radius: 0.25em;
                border: none;
                min-width: 6em;
            }
            QPushButton:hover {
                background-color: #3d4349;
            }
            QPushButton:pressed {
                background-color: #2d3238;
            }
        """)
        button_layout.addWidget(close_button)
        
        # Add to parent layout
        parent.layout().addLayout(button_layout)
        
    def _show_context_menu(self, position):
        """Show context menu for jobs tree"""
        item = self.jobs_tree.itemAt(position)
        if item:
            self.jobs_context_menu.exec_(self.jobs_tree.viewport().mapToGlobal(position))
    
    def _update_selected_job_progress(self, job):
        """Update progress display for selected job"""
        if hasattr(self, 'job_progress_bar'):
            progress, progress_text = selected_job_progress(job)
            self.job_progress_bar.setValue(progress)

            # Update progress label if exists
            if hasattr(self, 'progress_label'):
                self.progress_label.setText(progress_text)
        
    def _refresh_jobs_list(self):
        """Refresh the jobs list"""
        # Clear existing items
        self.jobs_tree.clear()
            
        # Add jobs
        for job_id, job in self.processor.jobs.items():
            # Progress, status, created, cost and source-file texts (shared with the mobile list)
            row = job_display_row(job_id, job)

            # Create tree widget item
            item = QTreeWidgetItem([
                row["display_id"],
                row["provider"],
                row["model"],  # Shorten model name
                row["status"],
                row["progress"],  # Now shows percentage and counts
                row["created"],
                row["source_file"],
                row["cost"]
            ])
            
            # Set color based on status (foreground + subtle background)
            fg_bg_map = {
                AsyncAPIStatus.PENDING: ("#e0a800", "#2a2412"),
                AsyncAPIStatus.PROCESSING: ("#4aa3ff", "#1b2938"),
                AsyncAPIStatus.COMPLETED: ("#5cb85c", "#1c2a1c"),
                AsyncAPIStatus.FAILED: ("#ff6b6b", "#2a1618"),
                AsyncAPIStatus.CANCELLED: ("#9ea3a8", "#242628"),
                AsyncAPIStatus.EXPIRED: ("#e0a800", "#2a2412")
            }
            fg, bg = fg_bg_map.get(job.status, ("#cfd3d8", "#1e1e1e"))
            for col in range(8):
                item.setForeground(col, QBrush(QColor(fg)))
                item.setBackground(col, QBrush(QColor(bg)))
            
            # Store job_id in item data for retrieval
            item.setData(0, Qt.UserRole, job_id)
            
            self.jobs_tree.addTopLevelItem(item)
        
        # Update progress bar if a job is selected
        if hasattr(self, 'selected_job_id') and self.selected_job_id:
            job = self.processor.jobs.get(self.selected_job_id)
            if job:
                self._update_selected_job_progress(job)

    def _get_selected_job_ids(self) -> List[str]:
        """Return list of job_ids for current selection"""
        selected_items = self.jobs_tree.selectedItems()
        job_ids = []
        for item in selected_items:
            job_id = item.data(0, Qt.UserRole)
            if job_id:
                job_ids.append(job_id)
        return job_ids
        
    def _on_job_select(self):
        """Handle job selection"""
        selected_items = self.jobs_tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            # Get full job ID from the item data
            job_id = item.data(0, Qt.UserRole)
            
            if job_id:
                self.selected_job_id = job_id
                
                # Update progress display for selected job
                job = self.processor.jobs.get(job_id)
                if job:
                    # Update progress bar if it exists
                    if hasattr(self, 'job_progress_bar'):
                        if job.total_requests > 0:
                            progress = int((job.completed_requests / job.total_requests) * 100)
                            self.job_progress_bar.setValue(progress)
                        else:
                            self.job_progress_bar.setValue(0)
                    
                    # Update progress label if it exists
                    if hasattr(self, 'progress_label'):
                        if job.total_requests > 0:
                            progress = int((job.completed_requests / job.total_requests) * 100)
                            self.progress_label.setText(
                                f"{progress}% ({job.completed_requests}/{job.total_requests} chapters)"
                            )
                        else:
                            self.progress_label.setText("0% (Waiting)")
                    
                    # Log selection
                    logger.info(f"Selected job: {job_id[:30]}... - Status: {job.status.value}")
                    
    def _start_auto_refresh(self, interval_seconds=30):
        """Start automatic status refresh"""
        def refresh():
            if hasattr(self, 'dialog') and self.dialog.isVisible():
                # Refresh all jobs
                refresh_pending_job_statuses(self.processor)

                self._refresh_jobs_list()
        
        # Create and start timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(refresh)
        self.refresh_timer.start(interval_seconds * 1000)  # Convert to milliseconds
        
        # Do first refresh immediately
        refresh()
    
    # ---- AsyncBatchJobMixin hooks: the Qt statements of the moved workflow methods ----
    _MB_OK = property(lambda self: QMessageBox.Ok)
    _MB_YES = property(lambda self: QMessageBox.Yes)
    _MB_NO = property(lambda self: QMessageBox.No)
    _MB_CANCEL = property(lambda self: QMessageBox.Cancel)

    def _async_msgbox(self, kind, *args):
        """``QMessageBox.<kind>(self.dialog, *args)``"""
        return getattr(QMessageBox, kind)(self.dialog, *args)

    def _async_single_shot(self, *args):
        """``QTimer.singleShot(*args)``"""
        return QTimer.singleShot(*args)

    def _async_process_events(self):
        """``QApplication.processEvents()``"""
        QApplication.processEvents()

    def _async_set_cost_info(self, text):
        """``self.cost_info_label.setText(text)``"""
        self.cost_info_label.setText(text)

    def _async_set_start_enabled(self, enabled):
        """``self.start_button.setEnabled(enabled)``"""
        self.start_button.setEnabled(enabled)

    def _async_wait_for_completion(self):
        """``self.wait_for_completion_checkbox.isChecked()``"""
        return self.wait_for_completion_checkbox.isChecked()

    def _async_poll_interval(self):
        """``self.poll_interval_spinbox.value()``"""
        return self.poll_interval_spinbox.value()

    def _async_set_wait_cursor(self, waiting):
        """``self.dialog.setCursor(Qt.WaitCursor / Qt.ArrowCursor)``"""
        self.dialog.setCursor(Qt.WaitCursor if waiting else Qt.ArrowCursor)

    def _async_dialog_visible(self):
        """``hasattr(self, 'dialog') and self.dialog.isVisible()``"""
        return hasattr(self, 'dialog') and self.dialog.isVisible()

    # Helper methods for thread-safe UI updates
    def _log(self, message, level="info"):
        """Thread-safe logging to GUI"""
        # Log based on level
        if level == "error":
            print(f"❌ {message}")  # This will show in GUI
        elif level == "warning":
            print(f"⚠️ {message}")  # This will show in GUI
        else:
            logger.info(message)  # This only goes to log file
            # Also display info messages in GUI
            if hasattr(self.gui, 'append_log'):
                QTimer.singleShot(0, lambda: self.gui.append_log(message))

    def _show_error(self, message):
        """Thread-safe error dialog"""
        self._log(f"Error: {message}", level="error")
        # Also show in the GUI log panel so it's visible even if dialog fails
        if hasattr(self.gui, 'append_log'):
            try:
                QTimer.singleShot(0, self.dialog, lambda: self.gui.append_log(f"❌ {message}"))
            except Exception:
                pass
        # Truncate for dialog display (very long messages can cause rendering issues)
        display_msg = message if len(message) <= 800 else message[:800] + '\n... (truncated)'
        try:
            QTimer.singleShot(0, self.dialog, lambda: QMessageBox.critical(self.dialog, "Error", display_msg))
        except Exception:
            pass

    def _show_info(self, title, message):
        """Thread-safe info dialog"""
        self._log(f"{title}: {message}", level="info")
        QTimer.singleShot(0, lambda: QMessageBox.information(self.dialog, title, message))

    def _show_warning(self, message):
        """Thread-safe warning display"""
        self._log(f"Warning: {message}", level="warning")

def _prewarm_dialog_offscreen(dialog):
    """Show a dialog offscreen at zero opacity once so Qt creates native resources."""
    if dialog is None:
        return
    def _center_dialog():
        try:
            screen = dialog.screen() or QApplication.primaryScreen()
            if screen is None:
                return
            geo = screen.availableGeometry()
            dialog.move(
                geo.x() + max(0, (geo.width() - dialog.width()) // 2),
                geo.y() + max(0, (geo.height() - dialog.height()) // 2),
            )
        except Exception:
            pass
    app = QApplication.instance()
    was_visible = dialog.isVisible()
    old_opacity = dialog.windowOpacity()
    try:
        dialog.setAttribute(Qt.WA_DontShowOnScreen, False)
        if not was_visible:
            dialog.setWindowOpacity(0.0)
            dialog.move(-20000, -20000)
            dialog.show()
            dialog._fade_native_window_seen = True
            dialog.raise_()
        try:
            dialog.ensurePolished()
            layout = dialog.layout()
            if layout is not None:
                layout.activate()
        except Exception:
            pass
        if app is not None:
            app.processEvents(QEventLoop.ExcludeUserInputEvents)
        if not was_visible:
            dialog.hide()
            _center_dialog()
            dialog.setWindowOpacity(old_opacity)
    except Exception:
        try:
            if not was_visible:
                dialog.hide()
                _center_dialog()
                dialog.setWindowOpacity(old_opacity)
        except Exception:
            pass


def show_async_processing_dialog(parent, translator_gui, show=True):
    """Show the async processing dialog
    
    Args:
        parent: Parent window (tkinter window - will be ignored for PySide6)
        translator_gui: Reference to main TranslatorGUI instance
    """
    # Reuse existing dialog if present to preserve state
    if hasattr(translator_gui, "async_dialog") and getattr(translator_gui, "async_dialog"):
        dlg_obj = translator_gui.async_dialog
        dlg_obj._refresh_model_info()
        if show:
            dlg_obj.dialog.setAttribute(Qt.WA_DontShowOnScreen, False)
            try:
                from dialog_animations import show_dialog_with_fade
                show_dialog_with_fade(dlg_obj.dialog, duration=180)
            except Exception:
                try:
                    dlg_obj.dialog.setWindowOpacity(1.0)
                except Exception:
                    pass
                dlg_obj.dialog.showNormal()
            dlg_obj.dialog.raise_()
            dlg_obj.dialog.activateWindow()
        else:
            _prewarm_dialog_offscreen(dlg_obj.dialog)
        return dlg_obj.dialog
    dlg_obj = AsyncProcessingDialog(parent, translator_gui)
    translator_gui.async_dialog = dlg_obj
    if show:
        dlg_obj.dialog.setAttribute(Qt.WA_DontShowOnScreen, False)
        try:
            from dialog_animations import show_dialog_with_fade
            show_dialog_with_fade(dlg_obj.dialog, duration=180)
        except Exception:
            dlg_obj.dialog.show()  # non-modal to allow hiding/restoring
    else:
        _prewarm_dialog_offscreen(dlg_obj.dialog)
    return dlg_obj.dialog


# Integration function for translator_gui.py
def add_async_processing_button(translator_gui, parent_frame):
    """Add async processing button to GUI
    
    This function should be called from translator_gui.py to add the button
    
    Args:
        translator_gui: TranslatorGUI instance
        parent_frame: Frame to add button to (PySide6 QWidget or layout)
    """
    # Create button with appropriate styling
    async_button = QPushButton("⚡ Async Processing (50% Off)")
    async_button.clicked.connect(lambda: show_async_processing_dialog(None, translator_gui))
    async_button.setStyleSheet("""
        QPushButton {
            background-color: #007bff;
            color: white;
            font-weight: bold;
            font-size: 11pt;
            padding: 0.5em 1em;
            border-radius: 0.25em;
            border: none;
        }
        QPushButton:hover {
            background-color: #0069d9;
        }
        QPushButton:pressed {
            background-color: #0056b3;
        }
    """)
    
    # Add to parent (assuming it's a layout)
    if hasattr(parent_frame, 'addWidget'):
        parent_frame.addWidget(async_button)
    
    # Store reference
    translator_gui.async_button = async_button
    
    return async_button
