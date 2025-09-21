# individual_endpoint_dialog.py
"""
Individual Endpoint Configuration Dialog for Glossarion
- Uses the application's WindowManager for consistent UI
- Allows enabling/disabling per-key custom endpoint (e.g., Azure, Ollama/local OpenAI-compatible)
- Persists changes to the in-memory key object and refreshes the parent list
"""
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from typing import Callable

try:
    # For type hints only; not required at runtime
    from multi_api_key_manager import APIKeyEntry  # noqa: F401
except Exception:
    pass


class IndividualEndpointDialog:
    def __init__(self, parent, translator_gui, key, refresh_callback: Callable[[], None], status_callback: Callable[[str], None]):
        self.parent = parent
        self.translator_gui = translator_gui
        self.key = key
        self.refresh_callback = refresh_callback
        self.status_callback = status_callback
        self.dialog = None
        self.canvas = None

        self._build()

    def _build(self):
        title = f"Configure Individual Endpoint — {getattr(self.key, 'model', '')}"

        if hasattr(self.translator_gui, 'wm'):
            # Use WindowManager scrollable dialog for consistency
            self.dialog, scrollable_frame, self.canvas = self.translator_gui.wm.setup_scrollable(
                self.parent,
                title,
                width=700,
                height=420,
                max_width_ratio=0.85,
                max_height_ratio=0.45
            )
        else:
            self.dialog = tk.Toplevel(self.parent)
            self.dialog.title(title)
            self.dialog.geometry("700x420")
            scrollable_frame = self.dialog

        main = tk.Frame(scrollable_frame, padx=20, pady=16)
        main.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 10))
        tk.Label(header, text="Per-Key Custom Endpoint", font=("TkDefaultFont", 14, "bold")).pack(side=tk.LEFT)

        # Enable toggle
        self.enable_var = tk.BooleanVar(value=bool(getattr(self.key, 'use_individual_endpoint', False)))
        tb.Checkbutton(header, text="Enable", variable=self.enable_var, bootstyle="round-toggle",
                       command=self._toggle_fields).pack(side=tk.RIGHT)

        # Description
        desc = (
            "Use a custom endpoint for this API key only. Works with OpenAI-compatible servers\n"
            "like Azure OpenAI or local providers (e.g., Ollama at http://localhost:11434/v1)."
        )
        tk.Label(main, text=desc, fg='gray', justify=tk.LEFT).pack(anchor=tk.W)

        # Form
        form = tk.LabelFrame(main, text="Endpoint Settings", padx=14, pady=12)
        form.pack(fill=tk.BOTH, expand=False, pady=(10, 0))

        # Endpoint URL
        tk.Label(form, text="Endpoint Base URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=6)
        self.endpoint_var = tk.StringVar(value=getattr(self.key, 'azure_endpoint', '') or '')
        self.endpoint_entry = tb.Entry(form, textvariable=self.endpoint_var)
        self.endpoint_entry.grid(row=0, column=1, sticky=tk.EW, pady=6)

        # Azure API version (optional; required if using Azure)
        tk.Label(form, text="Azure API Version:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=6)
        self.api_version_var = tk.StringVar(value=getattr(self.key, 'azure_api_version', '2025-01-01-preview') or '2025-01-01-preview')
        self.api_version_combo = ttk.Combobox(
            form,
            textvariable=self.api_version_var,
            values=[
                '2025-01-01-preview',
                '2024-12-01-preview',
                '2024-10-01-preview',
                '2024-08-01-preview',
                '2024-06-01',
                '2024-02-01',
                '2023-12-01-preview'
            ],
            width=24,
            state='readonly'
        )
        self.api_version_combo.grid(row=1, column=1, sticky=tk.W, pady=6)

        # Helper text
        hint = (
            "Hints:\n"
            "- Ollama: http://localhost:11434/v1\n"
            "- Azure OpenAI: https://<resource>.openai.azure.com/ (version required)\n"
            "- Other OpenAI-compatible: Provide the base URL ending with /v1 if applicable"
        )
        tk.Label(form, text=hint, fg='gray', justify=tk.LEFT, font=('TkDefaultFont', 9)).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0)
        )

        # Grid weights
        form.columnconfigure(1, weight=1)

        # Buttons
        btns = tk.Frame(main)
        btns.pack(fill=tk.X, pady=(14, 0))

        tb.Button(btns, text="Save", bootstyle="success", command=self._on_save).pack(side=tk.RIGHT)
        tb.Button(btns, text="Cancel", bootstyle="secondary", command=self._on_close).pack(side=tk.RIGHT, padx=(0, 8))
        tb.Button(btns, text="Disable", bootstyle="danger-outline", command=self._on_disable).pack(side=tk.LEFT)

        # Initial toggle state
        self._toggle_fields()

        # Window close protocol
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-size with WM if available
        if hasattr(self.translator_gui, 'wm') and self.canvas is not None:
            self.translator_gui.wm.auto_resize_dialog(self.dialog, self.canvas, max_width_ratio=0.9, max_height_ratio=0.45)

    def _toggle_fields(self):
        enabled = self.enable_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        self.endpoint_entry.config(state=state)
        # API version is only relevant for Azure but we leave it enabled while toggle is on
        self.api_version_combo.config(state='readonly' if enabled else 'disabled')

    def _is_azure_endpoint(self, url: str) -> bool:
        if not url:
            return False
        url_l = url.lower()
        return (".openai.azure.com" in url_l) or ("azure.com/openai" in url_l) or ("/openai/deployments/" in url_l)

    def _validate(self) -> bool:
        if not self.enable_var.get():
            return True
        url = (self.endpoint_var.get() or '').strip()
        if not url:
            messagebox.showerror("Validation Error", "Endpoint Base URL is required when Enable is ON.")
            return False
        if not (url.startswith("http://") or url.startswith("https://")):
            messagebox.showerror("Validation Error", "Endpoint URL must start with http:// or https://")
            return False
        if self._is_azure_endpoint(url):
            ver = (self.api_version_var.get() or '').strip()
            if not ver:
                messagebox.showerror("Validation Error", "Azure API Version is required for Azure endpoints.")
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
        enabled = self.enable_var.get()
        url = (self.endpoint_var.get() or '').strip()
        ver = (self.api_version_var.get() or '').strip()

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

        self.dialog.destroy()

    def _on_disable(self):
        # Disable quickly
        self.enable_var.set(False)
        self._toggle_fields()
        # Apply immediately and close
        self._on_save()

    def _on_close(self):
        self.dialog.destroy()
