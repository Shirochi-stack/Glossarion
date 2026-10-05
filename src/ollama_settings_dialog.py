"""Shared settings dialog for the local ``ollamapull/`` model route.

The dialog stores native Ollama request settings in the translator's normal
configuration. Network and installation work runs outside the Qt UI thread.
"""

from __future__ import annotations

import copy
import csv
import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
import threading
from io import StringIO
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea,
    QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)


# Pure settings helpers (route names, normalisation, option tables, parsers)
# live in ollama_settings (GUI-free, shared with mobile); re-exported here.
from ollama_settings import (
    COMMON_OPTION_KEYS,
    OLLAMAPULL_PREFIX,
    OPTION_GROUPS,
    RESERVED_REQUEST_KEYS,
    SLIDER_OPTIONS,
    _json_object,
    _reported_parameter_defaults,
    is_ollamapull_route,
    normalize_ollama_settings,
    ollama_settings_json,
    ollamapull_model_name,
    parse_option,
)


def _style_dropdown_arrow(combo: QComboBox) -> None:
    """Use the bundled Halgakos icon for this dialog's combo arrows."""
    icon_path = Path(__file__).resolve().with_name("Halgakos.ico")
    if not icon_path.is_file():
        return
    icon_url = icon_path.as_posix()
    combo.setStyleSheet(combo.styleSheet() + f"""
        QComboBox {{ padding-right: 24px; }}
        QComboBox::drop-down {{
            subcontrol-origin: padding; subcontrol-position: top right;
            width: 22px; border-left: 1px solid #485466;
        }}
        QComboBox::drop-down:disabled {{ border-left-color: #383d44; }}
        QComboBox::down-arrow {{ image: url(\"{icon_url}\"); width: 16px; height: 16px; }}
    """)


class _OptionEditor(QWidget):
    """Ollama option with an explicit model-default state."""

    def __init__(self, key: str, kind: str, saved=None, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.key = key
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.default = None
        if key == "draft_num_predict":
            self.mode = QComboBox()
            _style_dropdown_arrow(self.mode)
            self.mode.addItems(("Model default", "Off", "On (MTP if supported)"))
            self.mode.setCurrentIndex(0 if saved is None else 1 if int(saved) == 0 else 2)
            self._mode_help = (
                "Ollama chooses the draft method supported by the model. "
                "An embedded MTP model uses MTP; a model with a separate draft uses that draft."
            )
            self.mode.setToolTip(self._mode_help)
            self.control = QSpinBox()
            self.control.setRange(1, 2147483647)
            self.control.setValue(int(saved) if saved is not None and int(saved) > 0 else 4)
            self.control.setSuffix(" draft tokens")
            self.control.setMinimumWidth(155)
            self.mode.currentIndexChanged.connect(
                lambda index: self.control.setEnabled(index == 2)
            )
            self.control.setEnabled(self.mode.currentIndex() == 2)
            row.addWidget(self.mode, 1)
            row.addWidget(self.control)
        else:
            self.default = _styled_checkbox("Model default")
            self.default.setChecked(saved is None)
            self.default.setMinimumWidth(112)
            row.addWidget(self.default)

        if key == "draft_num_predict":
            pass
        elif kind == "bool" or key == "mirostat":
            self.control = QComboBox()
            _style_dropdown_arrow(self.control)
            choices = ("False", "True") if kind == "bool" else ("Off", "Mirostat 1", "Mirostat 2")
            self.control.addItems(choices)
            if saved is not None:
                self.control.setCurrentIndex(int(saved))
            row.addWidget(self.control, 1)
        elif key in ("num_ctx", "num_gpu", "num_thread"):
            self._reported_slider_default = None
            self.slider = QSlider(Qt.Horizontal)
            if key == "num_ctx":
                self.slider.setRange(1, max(262144, int(saved or 0)))
                self.slider.setSingleStep(256)
                self.slider.setPageStep(1024)
                suggested = 8192
                self.slider.setToolTip("Context size in tokens; drag to adjust.")
            elif key == "num_gpu":
                self.slider.setRange(-1, max(256, int(saved or 0)))
                self.slider.setPageStep(8)
                suggested = -1
                self.slider.setToolTip("-1 lets Ollama choose GPU layers automatically; 0 uses CPU only.")
            else:
                self.slider.setRange(0, max(os.cpu_count() or 1, int(saved or 0)))
                self.slider.setPageStep(4)
                suggested = 0
                self.slider.setToolTip("0 lets Ollama choose the thread count automatically.")
            self.slider.setValue(int(saved) if saved is not None else suggested)
            self.control = QLabel()
            self.control.setMinimumWidth(166)
            self.control.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.slider.valueChanged.connect(lambda _value: self._update_slider_label())
            self.default.toggled.connect(lambda _checked: self._update_slider_label())
            self._update_slider_label()
            row.addWidget(self.slider, 1)
            row.addWidget(self.control)
        elif key in SLIDER_OPTIONS:
            low, high, step, suggested = SLIDER_OPTIONS[key]
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, round((high - low) / step))
            self.control = QDoubleSpinBox()
            self.control.setDecimals(2 if step < 0.1 else 1)
            self.control.setSingleStep(step)
            self.control.setRange(-1000000, 1000000)
            self.control.setValue(float(saved) if saved is not None else suggested)
            self.control.setMinimumWidth(104)
            self.slider.setValue(max(0, min(self.slider.maximum(), round((self.control.value() - low) / step))))
            self.slider.valueChanged.connect(lambda value: self.control.setValue(low + value * step))
            def sync_slider(value):
                self.slider.blockSignals(True)
                self.slider.setValue(max(0, min(self.slider.maximum(), round((value - low) / step))))
                self.slider.blockSignals(False)
            self.control.valueChanged.connect(sync_slider)
            row.addWidget(self.slider, 1)
            row.addWidget(self.control)
        elif kind == "int":
            self.control = QSpinBox()
            self.control.setRange(-2147483647, 2147483647)
            self.control.setValue(int(saved) if saved is not None else 0)
            row.addWidget(self.control, 1)
        else:
            self.control = QLineEdit()
            if saved is not None:
                self.control.setText(json.dumps(saved) if kind == "array" else str(saved))
            self.control.setPlaceholderText("Enter JSON array" if kind == "array" else "Enter value")
            row.addWidget(self.control, 1)
        if self.default is not None:
            self.default.toggled.connect(lambda checked: self.control.setEnabled(not checked))
            if hasattr(self, "slider"):
                self.default.toggled.connect(lambda checked: self.slider.setEnabled(not checked))
                self.slider.setEnabled(not self.default.isChecked())
            self.control.setEnabled(not self.default.isChecked())
        self.setStyleSheet("""
            QLabel:disabled { color: #818892; }
            QSpinBox:disabled, QDoubleSpinBox:disabled,
            QLineEdit:disabled, QComboBox:disabled {
                color: #818892; background: #25272b; border-color: #383d44;
            }
            QSlider::groove:horizontal:disabled { background: #3a3f46; }
            QSlider::handle:horizontal:disabled { background: #747d87; }
        """)

    def text(self) -> str:
        if self.key == "draft_num_predict":
            return "" if self.mode.currentIndex() == 0 else "0" if self.mode.currentIndex() == 1 else str(self.control.value())
        if self.default.isChecked():
            return ""
        if self.kind == "bool":
            return "true" if self.control.currentIndex() else "false"
        if self.key == "mirostat":
            return str(self.control.currentIndex())
        if self.key in ("num_ctx", "num_gpu", "num_thread"):
            return str(self.slider.value())
        if isinstance(self.control, QLineEdit):
            return self.control.text()
        return str(self.control.value())

    def _update_slider_label(self) -> None:
        if self.default.isChecked():
            value = self._reported_slider_default
            fallback = "Model default" if self.key == "num_ctx" else "Model default: automatic"
            automatic = (self.key == "num_gpu" and value == -1) or (self.key == "num_thread" and value == 0)
            self.control.setText("Model default: automatic" if automatic else
                                 f"Model default: {value:,}" if value is not None else fallback)
        else:
            value = self.slider.value()
            automatic = (self.key == "num_gpu" and value == -1) or (self.key == "num_thread" and value == 0)
            self.control.setText("Automatic" if automatic
                                 else f"{value:,}")

    def setPlaceholderText(self, text: str) -> None:
        if self.key == "draft_num_predict":
            self.mode.setToolTip(text + "\n" + self._mode_help)
            return
        self.default.setToolTip(text)
        if self.key in ("num_ctx", "num_gpu", "num_thread") and text.startswith("Model default: "):
            try:
                value = int(text.removeprefix("Model default: ").strip())
                self._reported_slider_default = value
                self._update_slider_label()
            except ValueError:
                pass
        if isinstance(self.control, QLineEdit):
            self.control.setPlaceholderText(text)


def _size_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.1f} GiB"


def _hardware_status() -> dict[str, str]:
    """Best-effort local hardware snapshot; called only from a background job."""
    result = {"ram": "Unavailable", "gpu": "Unavailable", "vram": "Unavailable", "cpu": "Unavailable"}
    system = platform.system()
    cpu_name = platform.processor().strip()
    logical_cpus = os.cpu_count()
    if system == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or cpu_name
        except (ImportError, OSError):
            pass
        powershell = shutil.which("powershell.exe")
        if powershell and not cpu_name:
            try:
                process = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command",
                     "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                    capture_output=True, text=True, timeout=5, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if process.returncode == 0 and process.stdout.strip():
                    cpu_name = process.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
    elif system == "Darwin":
        try:
            process = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            if process.returncode == 0 and process.stdout.strip():
                cpu_name = process.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as stream:
                cpu_name = next((line.split(":", 1)[1].strip() for line in stream if line.startswith("model name")), cpu_name)
        except OSError:
            pass
    if cpu_name:
        result["cpu"] = cpu_name + (f" · {logical_cpus} logical CPUs" if logical_cpus else "")
    elif logical_cpus:
        result["cpu"] = f"{logical_cpus} logical CPUs"

    try:
        import psutil
        memory = psutil.virtual_memory()
        result["ram"] = f"{_size_gib(memory.total)} total · {_size_gib(memory.available)} available"
    except (ImportError, AttributeError, OSError):
        if system == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            memory = MemoryStatus()
            memory.dwLength = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                result["ram"] = f"{_size_gib(memory.ullTotalPhys)} total · {_size_gib(memory.ullAvailPhys)} available"
        elif system == "Linux":
            try:
                with open("/proc/meminfo", encoding="utf-8") as stream:
                    fields = {key.rstrip(":"): int(value.split()[0]) * 1024 for key, value in (line.split(":", 1) for line in stream if ":" in line)}
                result["ram"] = f"{_size_gib(fields['MemTotal'])} total · {_size_gib(fields['MemAvailable'])} available"
            except (OSError, KeyError, ValueError):
                pass
        elif system == "Darwin":
            try:
                process = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                    timeout=3, check=False,
                )
                if process.returncode == 0:
                    result["ram"] = f"{_size_gib(int(process.stdout.strip()))} total"
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            process = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            devices = list(csv.reader(StringIO(process.stdout))) if process.returncode == 0 else []
            if devices:
                result["gpu"] = "; ".join(row[0].strip() for row in devices if len(row) >= 3)
                result["vram"] = "; ".join(
                    f"{float(row[1]) / 1024:.1f} GiB total · {float(row[2]) / 1024:.1f} GiB free"
                    for row in devices if len(row) >= 3
                )
                return result
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if system == "Windows":
        powershell = shutil.which("powershell.exe")
        if powershell:
            try:
                process = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command",
                     "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
                    capture_output=True, text=True, timeout=6, check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                devices = json.loads(process.stdout) if process.returncode == 0 and process.stdout.strip() else []
                devices = devices if isinstance(devices, list) else [devices]
                names = [str(device.get("Name") or "").strip() for device in devices if isinstance(device, dict)]
                result["gpu"] = "; ".join(name for name in names if name) or result["gpu"]
                # Win32_VideoController.AdapterRAM is capped at 4 GiB on some
                # drivers, so do not present it as reliable installed VRAM.
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
    elif system == "Darwin":
        try:
            process = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            devices = json.loads(process.stdout).get("SPDisplaysDataType", []) if process.returncode == 0 else []
            if devices:
                result["gpu"] = "; ".join(
                    str(device.get("sppci_model") or device.get("_name") or "GPU")
                    for device in devices
                )
                vram = [str(device.get("spdisplays_vram") or device.get("spdisplays_vram_shared") or "") for device in devices]
                result["vram"] = "; ".join(value for value in vram if value) or "Shared system memory"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    elif system == "Linux":
        lspci = shutil.which("lspci")
        if lspci:
            try:
                process = subprocess.run([lspci], capture_output=True, text=True, timeout=4, check=False)
                names = [line.split(": ", 1)[-1] for line in process.stdout.splitlines()
                         if "VGA compatible controller" in line or "3D controller" in line]
                result["gpu"] = "; ".join(names) or result["gpu"]
            except (OSError, subprocess.SubprocessError):
                pass
    return result


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
            color: #818892;
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
        QLabel:disabled { color: #818892; }
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
                "🦙 Ollama Settings" if self._installed else "🦙 Load Ollama"
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
        for row, (key, label) in enumerate((
            ("ram", "System RAM"), ("vram", "GPU VRAM"),
            ("gpu", "GPU"), ("cpu", "CPU"),
        )):
            status_layout.addWidget(QLabel(label + ":"), row, 2)
            value = QLabel("Checking…")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.status_labels[key] = value
            status_layout.addWidget(value, row, 3)
        self.hardware_refresh_button = QPushButton("🔄")
        self.hardware_refresh_button.setToolTip("Refresh RAM and VRAM usage")
        self.hardware_refresh_button.setAccessibleName("Refresh hardware memory usage")
        self.hardware_refresh_button.setFixedSize(34, 30)
        self.hardware_refresh_button.clicked.connect(self.refresh_hardware_status)
        status_layout.addWidget(self.hardware_refresh_button, 0, 4, Qt.AlignTop)
        status_layout.setColumnStretch(1, 1)
        status_layout.setColumnStretch(3, 2)
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
        self.shutdown_button = QPushButton("Shutdown Ollama")
        self.shutdown_button.clicked.connect(self.shutdown_ollama)
        action_row.addWidget(self.shutdown_button)
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
        self.tabs.setObjectName("ollama_settings_tabs")
        self.tabs.setStyleSheet("""
            QTabWidget#ollama_settings_tabs::pane { border: 1px solid #485466; border-radius: 5px; background: #202124; }
            QTabWidget#ollama_settings_tabs QTabBar::tab {
                background: #292b30; color: #c6ccd5; border: 1px solid #485466;
                border-bottom: 0; padding: 10px 17px; margin-right: 3px;
                border-top-left-radius: 5px; border-top-right-radius: 5px;
            }
            QTabWidget#ollama_settings_tabs QTabBar::tab:selected {
                background: #334e69; color: #ffffff; border: 2px solid #69b8f4;
                border-bottom: 0; font-weight: bold;
            }
            QTabWidget#ollama_settings_tabs QTabBar::tab:hover:!selected:!disabled {
                background: #39414b; color: #ffffff;
            }
            QTabWidget#ollama_settings_tabs QTabBar::tab:disabled {
                background: #202124; color: #777d85; border-color: #363a40;
            }
            QTabWidget#ollama_settings_tabs QLabel:disabled,
            QTabWidget#ollama_settings_tabs QGroupBox:disabled {
                color: #818892;
            }
        """)
        layout.addWidget(self.tabs, 1)
        self.option_fields = {}
        options = self._model_settings.get("options")
        options = options if isinstance(options, dict) else {}
        for group_name, rows in OPTION_GROUPS:
            tab = QWidget()
            form = QFormLayout(tab)
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            for key, label, kind in rows:
                field = _OptionEditor(key, kind, options.get(key))
                field.setObjectName(f"ollama_option_{key}")
                if key == "draft_num_predict":
                    field.setToolTip(
                        "On sets draft_num_predict to the chosen token count. "
                        "Embedded MTP runs only when the model and Ollama support it. "
                        "Off sends 0; Model default sends no override."
                    )
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
        self.think_field = QComboBox()
        _style_dropdown_arrow(self.think_field)
        self.think_field.setEditable(True)
        self.think_field.addItems(["Model default", "True", "False", "low", "medium", "high"])
        self.think_field.setToolTip("Choose a thinking mode, or enter a model-supported level.")
        think = self._model_settings.get("think")
        if think is not None:
            self.think_field.setCurrentText(str(think).lower() if isinstance(think, bool) else str(think))
        advanced_form.addRow("Thinking:", self.think_field)
        self.keep_alive_field = QComboBox()
        _style_dropdown_arrow(self.keep_alive_field)
        self.keep_alive_field.setEditable(True)
        self.keep_alive_field.addItems(["Model default", "0", "5m", "30m", "1h", "-1"])
        if self._model_settings.get("keep_alive") is not None:
            self.keep_alive_field.setCurrentText(str(self._model_settings["keep_alive"]))
        advanced_form.addRow("Keep model loaded:", self.keep_alive_field)
        self.format_field = QComboBox()
        _style_dropdown_arrow(self.format_field)
        self.format_field.setEditable(True)
        self.format_field.addItems(["Model default", "json"])
        saved_format = self._model_settings.get("format")
        if saved_format is not None:
            self.format_field.setCurrentText(json.dumps(saved_format) if isinstance(saved_format, dict) else str(saved_format))
        self.format_field.setToolTip("Choose JSON or enter a JSON schema object.")
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
        footer.setSpacing(16)
        footer.addStretch()
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(170)
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        save_button = QPushButton("Save settings")
        save_button.setMinimumWidth(170)
        save_button.setDefault(True)
        save_button.clicked.connect(self.save_settings)
        footer.addWidget(save_button)
        footer.addStretch()
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
        self.shutdown_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)

    def refresh_status(self):
        self.refresh_button.setEnabled(False)
        self.progress_label.setText("Checking Ollama status…")

        def task(_progress):
            import ollamapull
            status = ollamapull.get_status(self.model_name)
            hardware = _hardware_status()
            details = None
            details_error = None
            if self.model_name and status.get("server_running") and status.get("model_installed"):
                try:
                    details = ollamapull.get_model_details(self.model_name)
                except Exception as exc:
                    details_error = str(exc)
            return {"status": status, "hardware": hardware, "details": details, "details_error": details_error}

        self._start_job("status", task)

    def refresh_hardware_status(self):
        self.hardware_refresh_button.setEnabled(False)

        def task(_progress):
            return _hardware_status()

        self._start_job("hardware", task)

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

    def shutdown_ollama(self):
        if self._busy_action:
            return
        answer = QMessageBox.question(
            self, "Shutdown Ollama",
            "Shut down the local Ollama server?\n\n"
            "This will interrupt active Ollama requests, including requests from other apps.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return
        self._set_action_busy(True)
        self.progress_label.setText("Shutting down Ollama…")

        def task(_progress):
            import ollamapull
            return ollamapull.shutdown_ollama()

        self._start_job("shutdown", task)

    def _on_job_progress(self, message):
        self.progress_label.setText(str(message))

    def _on_job_finished(self, payload):
        job_id, operation, result, error = payload
        self._jobs.pop(job_id, None)
        if operation == "hardware":
            self.hardware_refresh_button.setEnabled(True)
            if error:
                QMessageBox.warning(self, "Hardware status", f"Could not refresh memory usage:\n{error}")
            else:
                for key in ("ram", "vram", "gpu", "cpu"):
                    if key in result:
                        self.status_labels[key].setText(str(result[key]))
            return
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
            for key, value in (result.get("hardware") or {}).items():
                if key in ("ram", "vram", "gpu", "cpu"):
                    self.status_labels[key].setText(str(value))
            self.update_button.setText("Update Ollama" if status.get("update_available") else "Check / update Ollama")
            details = result.get("details")
            if details:
                self.model_info_edit.setPlainText(json.dumps(details, ensure_ascii=False, indent=2))
                defaults = _reported_parameter_defaults(details)
                for key, (field, _kind) in self.option_fields.items():
                    if key in defaults:
                        field.setPlaceholderText("Model default: " + defaults[key])
                model_info = details.get("model_info") if isinstance(details, dict) else None
                if isinstance(model_info, dict):
                    try:
                        layer_count = next(
                            int(value) for name, value in model_info.items()
                            if str(name).endswith(".block_count") and int(value) > 0
                        )
                        gpu_slider = self.option_fields["num_gpu"][0].slider
                        gpu_slider.setMaximum(max(layer_count + 1, gpu_slider.value()))
                    except (StopIteration, TypeError, ValueError):
                        pass
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
        self.progress_label.setText(
            "Ollama shut down." if operation == "shutdown" else
            "Ollama is ready." if operation == "ensure" else "Ollama update completed."
        )
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
        if "draft_num_predict" in options and options["draft_num_predict"] < 0:
            raise ValueError("draft_num_predict must be zero or greater")

        request = _json_object(self.extra_request_edit.toPlainText(), "Additional request fields")
        reserved = RESERVED_REQUEST_KEYS.intersection(request)
        if reserved:
            raise ValueError("Additional request fields cannot override: " + ", ".join(sorted(reserved)))
        result = copy.deepcopy(self._model_settings)
        result["options"] = options
        result["request"] = request

        think = self.think_field.currentText().strip()
        if think:
            if think.casefold() in ("default", "model default"):
                result.pop("think", None)
            elif think.casefold() in ("true", "false"):
                result["think"] = think.casefold() == "true"
            else:
                result["think"] = think
        else:
            result.pop("think", None)
        keep_alive = self.keep_alive_field.currentText().strip()
        if keep_alive and keep_alive.casefold() not in ("default", "model default"):
            result["keep_alive"] = keep_alive
        else:
            result.pop("keep_alive", None)
        response_format = self.format_field.currentText().strip()
        if response_format and response_format.casefold() not in ("default", "model default"):
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

