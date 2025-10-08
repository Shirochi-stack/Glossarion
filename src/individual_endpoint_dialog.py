# individual_endpoint_dialog.py
"""
Individual Endpoint Configuration Dialog for Glossarion
- Allows enabling/disabling per-key custom endpoint (e.g., Azure, Ollama/local OpenAI-compatible)
- Persists changes to the in-memory key object and refreshes the parent list
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QComboBox, QGroupBox, QGridLayout,
    QFrame, QScrollArea, QWidget, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from typing import Callable

try:
    # For type hints only; not required at runtime
    from multi_api_key_manager import APIKeyEntry  # noqa: F401
except Exception:
    pass


class IndividualEndpointDialog(QDialog):
    def __init__(self, parent, translator_gui, key, refresh_callback: Callable[[], None], status_callback: Callable[[str], None]):
        super().__init__(parent)
        self.translator_gui = translator_gui
        self.key = key
        self.refresh_callback = refresh_callback
        self.status_callback = status_callback
        
        self._build()

    def _build(self):
        title = f"Configure Individual Endpoint — {getattr(self.key, 'model', '')}"
        self.setWindowTitle(title)
        self.setMinimumSize(700, 420)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Per-Key Custom Endpoint")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_layout.addWidget(header_label)
        
        # Enable toggle
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(bool(getattr(self.key, 'use_individual_endpoint', False)))
        self.enable_checkbox.toggled.connect(self._toggle_fields)
        header_layout.addStretch()
        header_layout.addWidget(self.enable_checkbox)
        
        main_layout.addLayout(header_layout)
        
        # Description
        desc = (
            "Use a custom endpoint for this API key only. Works with OpenAI-compatible servers\n"
            "like Azure OpenAI or local providers (e.g., Ollama at http://localhost:11434/v1)."
        )
        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: gray;")
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Form group
        form_group = QGroupBox("Endpoint Settings")
        form_layout = QGridLayout(form_group)
        form_layout.setContentsMargins(14, 12, 14, 12)
        form_layout.setSpacing(6)
        
        # Endpoint URL
        endpoint_label = QLabel("Endpoint Base URL:")
        form_layout.addWidget(endpoint_label, 0, 0, Qt.AlignLeft)
        
        self.endpoint_entry = QLineEdit()
        self.endpoint_entry.setText(getattr(self.key, 'azure_endpoint', '') or '')
        form_layout.addWidget(self.endpoint_entry, 0, 1)
        
        # Azure API version
        api_version_label = QLabel("Azure API Version:")
        form_layout.addWidget(api_version_label, 1, 0, Qt.AlignLeft)
        
        self.api_version_combo = QComboBox()
        self.api_version_combo.addItems([
            '2025-01-01-preview',
            '2024-12-01-preview',
            '2024-10-01-preview',
            '2024-08-01-preview',
            '2024-06-01',
            '2024-02-01',
            '2023-12-01-preview'
        ])
        current_version = getattr(self.key, 'azure_api_version', '2025-01-01-preview') or '2025-01-01-preview'
        index = self.api_version_combo.findText(current_version)
        if index >= 0:
            self.api_version_combo.setCurrentIndex(index)
        form_layout.addWidget(self.api_version_combo, 1, 1)
        
        # Helper text
        hint = (
            "Hints:\n"
            "- Ollama: http://localhost:11434/v1\n"
            "- Azure OpenAI: https://<resource>.openai.azure.com/ (version required)\n"
            "- Other OpenAI-compatible: Provide the base URL ending with /v1 if applicable"
        )
        hint_label = QLabel(hint)
        hint_label.setStyleSheet("color: gray; font-size: 9pt;")
        hint_label.setWordWrap(True)
        form_layout.addWidget(hint_label, 2, 0, 1, 2, Qt.AlignLeft)
        
        # Make column 1 stretch
        form_layout.setColumnStretch(1, 1)
        
        main_layout.addWidget(form_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        disable_button = QPushButton("Disable")
        disable_button.clicked.connect(self._on_disable)
        button_layout.addWidget(disable_button)
        
        button_layout.addStretch()
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._on_close)
        button_layout.addWidget(cancel_button)
        
        save_button = QPushButton("Save")
        save_button.setDefault(True)
        save_button.clicked.connect(self._on_save)
        button_layout.addWidget(save_button)
        
        main_layout.addLayout(button_layout)
        
        # Initial toggle state
        self._toggle_fields()

    def _toggle_fields(self):
        enabled = self.enable_checkbox.isChecked()
        self.endpoint_entry.setEnabled(enabled)
        # API version is only relevant for Azure but we leave it enabled while toggle is on
        self.api_version_combo.setEnabled(enabled)

    def _is_azure_endpoint(self, url: str) -> bool:
        if not url:
            return False
        url_l = url.lower()
        return (".openai.azure.com" in url_l) or ("azure.com/openai" in url_l) or ("/openai/deployments/" in url_l)

    def _validate(self) -> bool:
        if not self.enable_checkbox.isChecked():
            return True
        url = self.endpoint_entry.text().strip()
        if not url:
            QMessageBox.critical(self, "Validation Error", "Endpoint Base URL is required when Enable is ON.")
            return False
        if not (url.startswith("http://") or url.startswith("https://")):
            QMessageBox.critical(self, "Validation Error", "Endpoint URL must start with http:// or https://")
            return False
        if self._is_azure_endpoint(url):
            ver = self.api_version_combo.currentText().strip()
            if not ver:
                QMessageBox.critical(self, "Validation Error", "Azure API Version is required for Azure endpoints.")
                return False
        return True

    def _persist_to_config_if_possible(self):
        """Best-effort persistence: update translator_gui.config['multi_api_keys'] for this key entry.
        We match by api_key and model to find the entry. If not found, skip silently.
        """
        try:
            cfg = getattr(self.translator_gui, 'config', None)
            if not isinstance(cfg, dict):
                return
            key_list = cfg.get('multi_api_keys', [])
            # Find by api_key AND model (best-effort)
            api_key = getattr(self.key, 'api_key', None)
            model = getattr(self.key, 'model', None)
            for entry in key_list:
                if entry.get('api_key') == api_key and entry.get('model') == model:
                    entry['use_individual_endpoint'] = bool(getattr(self.key, 'use_individual_endpoint', False))
                    entry['azure_endpoint'] = getattr(self.key, 'azure_endpoint', None)
                    entry['azure_api_version'] = getattr(self.key, 'azure_api_version', None)
                    break
            # Save without message
            if hasattr(self.translator_gui, 'save_config'):
                self.translator_gui.save_config(show_message=False)
        except Exception:
            # Non-fatal
            pass

    def _on_save(self):
        if not self._validate():
            return
        enabled = self.enable_checkbox.isChecked()
        url = self.endpoint_entry.text().strip()
        ver = self.api_version_combo.currentText().strip()

        # Apply to key object
        self.key.use_individual_endpoint = enabled
        self.key.azure_endpoint = url if enabled else None
        # Keep API version even if disabled, but it's only used when enabled
        self.key.azure_api_version = ver or getattr(self.key, 'azure_api_version', '2025-01-01-preview')

        # Notify parent UI
        if callable(self.refresh_callback):
            try:
                self.refresh_callback()
            except Exception:
                pass
        if callable(self.status_callback):
            try:
                if enabled and url:
                    self.status_callback(f"Individual endpoint set: {url}")
                else:
                    self.status_callback("Individual endpoint disabled")
            except Exception:
                pass

        # Best-effort persistence to config
        self._persist_to_config_if_possible()

        self.accept()

    def _on_disable(self):
        # Disable quickly
        self.enable_checkbox.setChecked(False)
        self._toggle_fields()
        # Apply immediately and close
        self._on_save()

    def _on_close(self):
        self.reject()
