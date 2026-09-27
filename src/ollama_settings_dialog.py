"""Shared settings dialog for the local ``ollamapull/`` model route.

The dialog stores native Ollama request settings in the translator's normal
configuration. Network and installation work runs outside the Qt UI thread.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget,
)


OLLAMAPULL_PREFIX = "ollamapull/"


def is_ollamapull_route(value: str) -> bool:
    """Recognize the route while a user is still typing its model name."""
    value = str(value or "").strip().casefold()
    return value == "ollamapull" or value.startswith(OLLAMAPULL_PREFIX)


def ollamapull_model_name(value: str) -> str:
    """Return the native Ollama model name, or an empty string for other routes."""
    value = str(value or "").strip()
    if not value.casefold().startswith(OLLAMAPULL_PREFIX):
        return ""
    return value[len(OLLAMAPULL_PREFIX):].strip()


def normalize_ollama_settings(value) -> dict:
    """Preserve future settings while supplying the defaults used by the UI."""
    settings = copy.deepcopy(value) if isinstance(value, dict) else {}
    settings.setdefault("auto_update", True)
    if not isinstance(settings.get("models"), dict):
        settings["models"] = {}
    return settings


def ollama_settings_json(config: dict) -> str:
    """Serialize the shared settings for translation workers and key tests."""
    return json.dumps(
        normalize_ollama_settings((config or {}).get("ollama_settings")),
        ensure_ascii=False,
        separators=(",", ":"),
    )


# Ollama's published Modelfile parameters, plus native request options useful
# for runtime tuning. Unlisted future options remain available in Advanced.
OPTION_GROUPS = (
    ("Context and output", (
        ("num_ctx", "Context size", "int"),
        ("num_predict", "Maximum generated tokens", "int"),
        ("draft_num_predict", "MTP draft tokens", "int"),
        ("num_keep", "Prompt tokens to retain", "int"),
        ("num_batch", "Prompt batch size", "int"),
    )),
    ("Sampling", (
        ("temperature", "Temperature", "float"),
        ("top_k", "Top K", "int"),
        ("top_p", "Top P", "float"),
        ("min_p", "Min P", "float"),
        ("typical_p", "Typical P", "float"),
        ("tfs_z", "Tail free sampling", "float"),
        ("repeat_last_n", "Repeat lookback", "int"),
        ("repeat_penalty", "Repeat penalty", "float"),
        ("presence_penalty", "Presence penalty", "float"),
        ("frequency_penalty", "Frequency penalty", "float"),
        ("penalize_newline", "Penalize newline", "bool"),
        ("mirostat", "Mirostat mode", "int"),
        ("mirostat_tau", "Mirostat target", "float"),
        ("mirostat_eta", "Mirostat learning rate", "float"),
        ("seed", "Random seed", "int"),
        ("stop", "Stop sequences (JSON array)", "array"),
    )),
    ("Runtime", (
        ("num_gpu", "GPU layers", "int"),
        ("main_gpu", "Primary GPU", "int"),
        ("num_thread", "CPU threads", "int"),
        ("low_vram", "Low VRAM mode", "bool"),
        ("use_mmap", "Memory mapping", "bool"),
        ("use_mlock", "Lock model in memory", "bool"),
        ("numa", "NUMA mode", "bool"),
        ("vocab_only", "Vocabulary only", "bool"),
    )),
)
COMMON_OPTION_KEYS = {
    name for _group, rows in OPTION_GROUPS for name, _label, _kind in rows
}
RESERVED_REQUEST_KEYS = {"model", "messages", "stream", "options", "think", "keep_alive", "format"}


def parse_option(text: str, kind: str):
    """Parse a setting without silently changing what will reach Ollama."""
    value = text.strip()
    if kind == "int":
        return int(value)
    if kind == "float":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("must be a finite number")
        return number
    if kind == "bool":
        if value.casefold() in ("true", "1", "yes", "on"):
            return True
        if value.casefold() in ("false", "0", "no", "off"):
            return False
        raise ValueError("enter true or false")
    if kind == "array":
        result = json.loads(value)
        if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
            raise ValueError("enter a JSON array of strings")
        return result
    raise ValueError(f"unknown option type: {kind}")


def _json_object(text: str, label: str) -> dict:
    value = json.loads(text.strip() or "{}")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _reported_parameter_defaults(details: dict) -> dict:
    """Read the defaults Ollama reports in /api/show's parameters block."""
    defaults = {}
    parameters = details.get("parameters", "") if isinstance(details, dict) else ""
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            line = line.strip()
            if line.casefold().startswith("parameter "):
                line = line[10:].strip()
            name, separator, value = line.partition(" ")
            if separator and name:
                defaults[name] = value.strip()
    return defaults


def _styled_checkbox(text: str) -> QCheckBox:
    """Match Glossarion's blue checkbox with its visible white checkmark."""
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
    checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
    checkmark.setGeometry(2, 1, 14, 14)

    def update_checkmark():
        checkmark.setVisible(checkbox.isChecked())
        if checkbox.isChecked():
            checkmark.raise_()

    checkbox.stateChanged.connect(update_checkmark)
    checkbox._checkmark_label = checkmark
    checkbox._update_checkmark = update_checkmark
    update_checkmark()
    return checkbox


class _JobSignals(QObject):
    progress = Signal(str)
    finished = Signal(object)


class _OllamaJob(QRunnable):
    def __init__(self, operation, function):
        super().__init__()
        self.operation = operation
        self.function = function
        self.signals = _JobSignals()

    def run(self):
        try:
            result = self.function(self.signals.progress.emit)
            self.signals.finished.emit((id(self), self.operation, result, None))
        except Exception as exc:
            self.signals.finished.emit((id(self), self.operation, None, str(exc)))


class OllamaRouteButtonController(QObject):
    """Bind one model field to a non-blocking install/settings button."""

    def __init__(self, button, translator_gui, model_text, dialog_parent=None):
        super().__init__(button)
        self.button = button
        self.translator_gui = translator_gui
        self.model_text = model_text
        self.dialog_parent = dialog_parent or translator_gui
        self._installed = None
        self._busy = False
        self._passive_check_pending = False
        self._jobs = {}
        self._stop_event = threading.Event()
        self._progress_dialog = None
        button.clicked.connect(self._on_clicked)

    def update_model(self, text=None):
        if text is None:
            text = self.model_text()
        active = is_ollamapull_route(text)
        self.button.setVisible(active)
        if not active:
            return
        if not self._busy:
            self.button.setText(
                "🦙 Ollama Settings" if self._installed else "🦙 Download Ollama"
            )
        if self._installed is None and not self._passive_check_pending and not self._busy:
            self._passive_check_pending = True
            QTimer.singleShot(0, self._check_installation_passively)

    def _start_job(self, operation, function):
        job = _OllamaJob(operation, function)
        self._jobs[id(job)] = job
        job.signals.progress.connect(self._on_progress)
        job.signals.finished.connect(self._on_job_finished)
        QThreadPool.globalInstance().start(job)

    def _check_installation_passively(self):
        if not is_ollamapull_route(self.model_text()):
            self._passive_check_pending = False
            return

        def task(_progress):
            import ollamapull
            return ollamapull.get_status("")

        self._start_job("passive_status", task)

    def _on_clicked(self):
        if self._busy or not is_ollamapull_route(self.model_text()):
            return
        self._busy = True
        self.button.setEnabled(False)
        self.button.setText("Checking Ollama…")

        def task(_progress):
            import ollamapull
            return ollamapull.get_status("")

        self._start_job("click_status", task)

    def _begin_install(self):
        self._stop_event.clear()
        self.button.setText("Installing Ollama…")
        progress_dialog = QProgressDialog(
            "Installing Ollama…", "Cancel", 0, 0, self.dialog_parent,
        )
        progress_dialog.setWindowTitle("Download Ollama")
        progress_dialog.setWindowModality(Qt.NonModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.canceled.connect(self._stop_event.set)
        self._progress_dialog = progress_dialog
        progress_dialog.show()
        auto_update = bool(normalize_ollama_settings(
            self.translator_gui.config.get("ollama_settings")
        ).get("auto_update", True))

        def task(progress):
            import ollamapull
            try:
                ollamapull.ensure_ready(
                    None, auto_update=auto_update,
                    progress=progress, should_stop=self._stop_event.is_set,
                )
                return ollamapull.get_status("")
            except Exception as install_error:
                # An installer may finish successfully while its own serve
                # subprocess exits because the desktop Ollama server started.
                # Only this startup race can be cleared by a healthy server.
                # Cancellation and unrelated install/update errors still matter.
                message = str(install_error).casefold()
                startup_error = (
                    message.startswith("ollama server exited")
                    or message.startswith("ollama did not start within")
                )
                if (
                    self._stop_event.is_set()
                    or isinstance(install_error, ollamapull.OllamaPullCancelled)
                    or not startup_error
                ):
                    raise
                try:
                    status = ollamapull.get_status("")
                except Exception:
                    raise install_error
                if status.get("installed") and status.get("server_running"):
                    return status
                raise

        self._start_job("install", task)

    def _on_progress(self, message):
        if self._progress_dialog is not None:
            self._progress_dialog.setLabelText(str(message))

    def _finish_click(self):
        self._busy = False
        self.button.setEnabled(True)
        self.update_model()

    def _on_job_finished(self, payload):
        job_id, operation, result, error = payload
        self._jobs.pop(job_id, None)
        if operation == "passive_status":
            self._passive_check_pending = False
            if not self._busy:
                self._installed = False if error else bool((result or {}).get("installed"))
            self.update_model()
            return

        if operation == "click_status":
            if error:
                self._installed = False
                self._finish_click()
                QMessageBox.warning(
                    self.dialog_parent, "Ollama status",
                    f"Could not check Ollama installation:\n{error}",
                )
                return
            self._installed = bool((result or {}).get("installed"))
            if not is_ollamapull_route(self.model_text()):
                self._finish_click()
                return
            if self._installed:
                self._finish_click()
                model = self.model_text()
                if is_ollamapull_route(model):
                    open_ollama_settings(self.translator_gui, model, self.dialog_parent)
            else:
                self._begin_install()
            return

        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        if error:
            self._installed = False
            self._finish_click()
            QMessageBox.warning(
                self.dialog_parent, "Download Ollama",
                f"Ollama installation could not complete:\n{error}",
            )
            return
        self._installed = bool((result or {}).get("installed"))
        self._finish_click()
        if not self._installed:
            QMessageBox.warning(
                self.dialog_parent, "Download Ollama",
                "The installer finished, but Ollama was not found. Refresh its status or retry the download.",
            )


class OllamaSettingsDialog(QDialog):
    """Edit native Ollama settings for one ``ollamapull/`` model."""

    def __init__(self, translator_gui, routed_model: str, parent=None):
        super().__init__(parent or translator_gui)
        self.translator_gui = translator_gui
        self.model_name = ollamapull_model_name(routed_model)
        if not is_ollamapull_route(routed_model):
            raise ValueError("Ollama settings require an ollamapull route")
        self._settings = normalize_ollama_settings(
            translator_gui.config.get("ollama_settings")
        )
        self._model_settings = copy.deepcopy(
            self._settings["models"].get(self.model_name, {})
        )
        if not isinstance(self._model_settings, dict):
            self._model_settings = {}
        self._jobs = {}
        self._busy_action = False
        self._stop_event = threading.Event()
        self.setWindowTitle(
            f"Ollama Settings · {self.model_name}" if self.model_name else "Ollama Settings"
        )
        self.setMinimumSize(750, 620)
        self.resize(850, 720)
        self._build_ui()
        QTimer.singleShot(0, self.refresh_status)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        heading = QLabel(
            f"Local Ollama model: {self.model_name}" if self.model_name else "Local Ollama"
        )
        heading.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(heading)

        status_box = QGroupBox("Ollama status")
        status_layout = QGridLayout(status_box)
        self.status_labels = {}
        for row, (key, label) in enumerate((
            ("installed", "Installation"), ("installed_version", "Installed version"),
            ("server_version", "Running server version"),
            ("latest_version", "Latest version"), ("server_running", "Server"),
            ("restart_required", "Restart needed"),
            ("model_installed", "Selected model downloaded"),
            ("model_loaded", "Selected model in memory"),
        )):
            status_layout.addWidget(QLabel(label + ":"), row, 0)
            value = QLabel("Checking…")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if key == "model_loaded":
                value.setToolTip("Loaded in memory does not show whether a chat request is generating tokens.")
            self.status_labels[key] = value
            status_layout.addWidget(value, row, 1)
        layout.addWidget(status_box)

        action_row = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh status")
        self.refresh_button.clicked.connect(self.refresh_status)
        action_row.addWidget(self.refresh_button)
        self.ensure_button = QPushButton(
            "Install / start / pull model" if self.model_name else "Install / start Ollama"
        )
        self.ensure_button.clicked.connect(self.ensure_model_ready)
        action_row.addWidget(self.ensure_button)
        self.update_button = QPushButton("Update Ollama")
        self.update_button.clicked.connect(self.update_ollama)
        action_row.addWidget(self.update_button)
        self.cancel_button = QPushButton("Stop operation")
        self.cancel_button.clicked.connect(self._stop_event.set)
        self.cancel_button.hide()
        action_row.addWidget(self.cancel_button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self.progress_label = QLabel(" ")
        self.progress_label.setWordWrap(True)
        self.progress_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.progress_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.option_fields = {}
        options = self._model_settings.get("options")
        options = options if isinstance(options, dict) else {}
        for group_name, rows in OPTION_GROUPS:
            tab = QWidget()
            form = QFormLayout(tab)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            for key, label, kind in rows:
                field = QLineEdit()
                field.setObjectName(f"ollama_option_{key}")
                field.setPlaceholderText("Use model default" if kind != "bool" else "Default, true, or false")
                if key in options:
                    field.setText(json.dumps(options[key]) if kind == "array" else str(options[key]).lower() if kind == "bool" else str(options[key]))
                if key == "draft_num_predict":
                    field.setToolTip("Multi-token prediction draft length. Availability depends on the installed Ollama version and model.")
                form.addRow(label + ":", field)
                self.option_fields[key] = (field, kind)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab)
            self.tabs.addTab(scroll, group_name)

        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_form = QFormLayout()
        self.auto_update_checkbox = _styled_checkbox("Auto-update Ollama before use")
        self.auto_update_checkbox.setChecked(bool(self._settings.get("auto_update", True)))
        advanced_form.addRow(self.auto_update_checkbox)
        self.think_field = QLineEdit()
        self.think_field.setPlaceholderText("Default, true, false, or a model-supported level")
        think = self._model_settings.get("think")
        if think is not None:
            self.think_field.setText(str(think).lower() if isinstance(think, bool) else str(think))
        advanced_form.addRow("Thinking:", self.think_field)
        self.keep_alive_field = QLineEdit(str(self._model_settings.get("keep_alive", "")))
        self.keep_alive_field.setPlaceholderText("Default, 5m, 0, …")
        advanced_form.addRow("Keep model loaded:", self.keep_alive_field)
        self.format_field = QLineEdit()
        saved_format = self._model_settings.get("format")
        if saved_format is not None:
            self.format_field.setText(json.dumps(saved_format) if isinstance(saved_format, dict) else str(saved_format))
        self.format_field.setPlaceholderText("Default, json, or a JSON schema object")
        advanced_form.addRow("Response format:", self.format_field)
        advanced_layout.addLayout(advanced_form)

        advanced_layout.addWidget(QLabel("Additional Ollama options (JSON object):"))
        self.extra_options_edit = QPlainTextEdit()
        self.extra_options_edit.setObjectName("ollama_extra_options")
        self.extra_options_edit.setPlaceholderText('{"option_name": value}')
        self.extra_options_edit.setPlainText(json.dumps(
            {key: value for key, value in options.items() if key not in COMMON_OPTION_KEYS},
            ensure_ascii=False, indent=2,
        ))
        advanced_layout.addWidget(self.extra_options_edit, 1)
        advanced_layout.addWidget(QLabel("Additional /api/chat fields (JSON object):"))
        self.extra_request_edit = QPlainTextEdit()
        self.extra_request_edit.setObjectName("ollama_extra_request")
        self.extra_request_edit.setPlaceholderText('{"logprobs": true}')
        self.extra_request_edit.setPlainText(json.dumps(
            self._model_settings.get("request") if isinstance(self._model_settings.get("request"), dict) else {},
            ensure_ascii=False, indent=2,
        ))
        advanced_layout.addWidget(self.extra_request_edit, 1)
        self.tabs.addTab(advanced, "Advanced")

        info = QWidget()
        info_layout = QVBoxLayout(info)
        self.model_defaults_label = QLabel("Model defaults: checking…")
        self.model_defaults_label.setWordWrap(True)
        info_layout.addWidget(self.model_defaults_label)
        info_layout.addWidget(QLabel("Model metadata and defaults reported by Ollama (/api/show):"))
        self.model_info_edit = QPlainTextEdit()
        self.model_info_edit.setReadOnly(True)
        self.model_info_edit.setPlainText("Checking local model…")
        info_layout.addWidget(self.model_info_edit)
        self.tabs.addTab(info, "Model details")
        if not self.model_name:
            for index in range(3):
                self.tabs.setTabEnabled(index, False)
            for field in (self.think_field, self.keep_alive_field, self.format_field,
                          self.extra_options_edit, self.extra_request_edit):
                field.setEnabled(False)
            self.model_info_edit.setPlainText(
                "Enter ollamapull/model-name in a model field to configure that model."
            )
            self.model_defaults_label.setText("No model selected.")

        footer = QHBoxLayout()
        footer.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        save_button = QPushButton("Save settings")
        save_button.setDefault(True)
        save_button.clicked.connect(self.save_settings)
        footer.addWidget(save_button)
        layout.addLayout(footer)

    def _start_job(self, operation, function):
        job = _OllamaJob(operation, function)
        # Retain the Python wrapper and its signals through queued delivery.
        self._jobs[id(job)] = job
        job.signals.progress.connect(self._on_job_progress)
        job.signals.finished.connect(self._on_job_finished)
        QThreadPool.globalInstance().start(job)

    def _set_action_busy(self, busy):
        self._busy_action = busy
        self.ensure_button.setEnabled(not busy)
        self.update_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)

    def refresh_status(self):
        self.refresh_button.setEnabled(False)
        self.progress_label.setText("Checking Ollama status…")

        def task(_progress):
            import ollamapull
            status = ollamapull.get_status(self.model_name)
            details = None
            details_error = None
            if self.model_name and status.get("server_running") and status.get("model_installed"):
                try:
                    details = ollamapull.get_model_details(self.model_name)
                except Exception as exc:
                    details_error = str(exc)
            return {"status": status, "details": details, "details_error": details_error}

        self._start_job("status", task)

    def ensure_model_ready(self):
        if self._busy_action:
            return
        self._stop_event.clear()
        self._set_action_busy(True)
        self.progress_label.setText("Preparing Ollama and model…")
        auto_update = self.auto_update_checkbox.isChecked()

        def task(progress):
            import ollamapull
            ollamapull.ensure_ready(
                self.model_name or None, auto_update=auto_update,
                progress=progress, should_stop=self._stop_event.is_set,
            )
            try:
                from model_options import refresh_ollamapull_model_catalog
                refresh_ollamapull_model_catalog()
            except Exception:
                pass
            return True

        self._start_job("ensure", task)

    def update_ollama(self):
        if self._busy_action:
            return
        self._stop_event.clear()
        self._set_action_busy(True)
        self.progress_label.setText("Updating Ollama…")

        def task(progress):
            import ollamapull
            ollamapull.update_ollama(progress=progress, should_stop=self._stop_event.is_set)
            return True

        self._start_job("update", task)

    def _on_job_progress(self, message):
        self.progress_label.setText(str(message))

    def _on_job_finished(self, payload):
        job_id, operation, result, error = payload
        self._jobs.pop(job_id, None)
        if operation == "status":
            self.refresh_button.setEnabled(True)
            if error:
                self.progress_label.setText(f"Status check failed: {error}")
                self.model_info_edit.setPlainText(f"Could not read model details: {error}")
                return
            status = result.get("status") or {}
            self.status_labels["installed"].setText("Installed" if status.get("installed") else "Not installed")
            self.status_labels["installed_version"].setText(str(status.get("installed_version") or status.get("version") or "Unknown"))
            self.status_labels["server_version"].setText(str(status.get("server_version") or "Unknown"))
            self.status_labels["latest_version"].setText(str(status.get("latest_version") or "Unknown"))
            self.status_labels["server_running"].setText("Running" if status.get("server_running") else "Stopped")
            self.status_labels["restart_required"].setText(
                "Yes — restart Ollama to use the update" if status.get("restart_required") else "No"
            )
            self.status_labels["model_installed"].setText(
                "No model selected" if not self.model_name else
                "Server stopped" if not status.get("server_running") else
                "Downloaded" if status.get("model_installed") is True else
                "Not downloaded" if status.get("model_installed") is False else "Unknown"
            )
            self.status_labels["model_loaded"].setText(
                "No model selected" if not self.model_name else
                "Server stopped" if not status.get("server_running") else
                "Loaded" if status.get("model_loaded") is True else
                "Not loaded" if status.get("model_loaded") is False else "Unknown"
            )
            self.update_button.setText("Update Ollama" if status.get("update_available") else "Check / update Ollama")
            details = result.get("details")
            if details:
                self.model_info_edit.setPlainText(json.dumps(details, ensure_ascii=False, indent=2))
                defaults = _reported_parameter_defaults(details)
                for key, (field, _kind) in self.option_fields.items():
                    if key in defaults:
                        field.setPlaceholderText("Model default: " + defaults[key])
                summary = ", ".join(
                    f"{key}={value}" for key, value in defaults.items()
                ) if defaults else "no explicit parameters reported"
                self.model_defaults_label.setText("Model defaults: " + summary[:500])
            elif result.get("details_error"):
                self.model_info_edit.setPlainText("Could not read model details: " + result["details_error"])
                self.model_defaults_label.setText("Model defaults unavailable.")
            elif self.model_name:
                self.model_info_edit.setPlainText("Download the model to view its metadata and defaults.")
                self.model_defaults_label.setText("Model defaults unavailable until download.")
            self.progress_label.setText(
                "Update installed. Restart Ollama to use it."
                if status.get("restart_required") else "Status is current."
            )
            return

        self._set_action_busy(False)
        if error:
            self.progress_label.setText(f"Ollama {operation} failed: {error}")
            QMessageBox.warning(self, "Ollama", f"Ollama {operation} failed:\n{error}")
            return
        self.progress_label.setText("Ollama is ready." if operation == "ensure" else "Ollama update completed.")
        if operation == "ensure":
            try:
                refresh = getattr(self.translator_gui, "_refresh_model_combo_catalog", None)
                if callable(refresh):
                    from model_options import get_model_options, merge_saved_model_options
                    config = self.translator_gui.config
                    models = merge_saved_model_options(
                        config.get("custom_model_list"), get_model_options(),
                        config.get("model_manager_removed_models", []),
                    )
                    refresh(models)
            except Exception:
                pass
        self.refresh_status()

    def _collect_model_settings(self):
        extra_options = _json_object(self.extra_options_edit.toPlainText(), "Additional options")
        duplicates = COMMON_OPTION_KEYS.intersection(extra_options)
        if duplicates:
            raise ValueError("Additional options duplicate a named field: " + ", ".join(sorted(duplicates)))
        options = dict(extra_options)
        for key, (field, kind) in self.option_fields.items():
            text = field.text().strip()
            if text:
                try:
                    options[key] = parse_option(text, kind)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{key}: {exc}") from exc
        if "num_ctx" in options and options["num_ctx"] <= 0:
            raise ValueError("num_ctx must be greater than zero")

        request = _json_object(self.extra_request_edit.toPlainText(), "Additional request fields")
        reserved = RESERVED_REQUEST_KEYS.intersection(request)
        if reserved:
            raise ValueError("Additional request fields cannot override: " + ", ".join(sorted(reserved)))
        result = copy.deepcopy(self._model_settings)
        result["options"] = options
        result["request"] = request

        think = self.think_field.text().strip()
        if think:
            if think.casefold() == "default":
                result.pop("think", None)
            elif think.casefold() in ("true", "false"):
                result["think"] = think.casefold() == "true"
            else:
                result["think"] = think
        else:
            result.pop("think", None)
        keep_alive = self.keep_alive_field.text().strip()
        if keep_alive and keep_alive.casefold() != "default":
            result["keep_alive"] = keep_alive
        else:
            result.pop("keep_alive", None)
        response_format = self.format_field.text().strip()
        if response_format and response_format.casefold() != "default":
            if response_format.startswith("{"):
                result["format"] = _json_object(response_format, "Response format")
            elif response_format == "json":
                result["format"] = "json"
            else:
                raise ValueError("Response format must be json or a JSON schema object")
        else:
            result.pop("format", None)
        return result

    def save_settings(self):
        try:
            model_settings = self._collect_model_settings()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Invalid Ollama settings", str(exc))
            return
        settings = normalize_ollama_settings(
            self.translator_gui.config.get("ollama_settings")
        )
        settings["auto_update"] = self.auto_update_checkbox.isChecked()
        if self.model_name:
            settings["models"][self.model_name] = model_settings
        self.translator_gui.config["ollama_settings"] = settings
        os.environ["OLLAMA_SETTINGS_JSON"] = ollama_settings_json(self.translator_gui.config)
        save = getattr(self.translator_gui, "save_config", None)
        if callable(save):
            saved = save(show_message=False)
            if saved is False:
                QMessageBox.warning(self, "Ollama settings", "Settings could not be saved to the Glossarion configuration.")
                return
        self.accept()

    def reject(self):
        self._stop_event.set()
        super().reject()


def open_ollama_settings(translator_gui, routed_model: str, parent=None):
    """Open the shared dialog for either the translator or key manager."""
    if not is_ollamapull_route(routed_model):
        return None
    dialog = OllamaSettingsDialog(translator_gui, routed_model, parent)
    return dialog.exec_()

