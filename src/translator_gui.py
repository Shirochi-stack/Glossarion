import io
import os
import sys
import json
import threading
import subprocess
import math
import ttkbootstrap as tb
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from ttkbootstrap.constants import *
import logging
import shutil

if getattr(sys, "frozen", False) and len(sys.argv) > 1:
    # first arg is our helper name
    helper, *rest = sys.argv[1:]
    if helper == "TransateKRtoEN":
        from TransateKRtoEN import main as _main
        sys.exit(_main(*rest))
    elif helper == "extract_glossary_from_epub":
        from extract_glossary_from_epub import main as _main
        sys.exit(_main(*rest))
    elif helper == "epub_fallback_compiler_with_cover_portable":
        from epub_fallback_compiler_with_cover_portable import main as _main
        sys.exit(_main(*rest))
        
def build_subprocess_args(script_basename, *params):
    exe = sys.executable
    if getattr(sys, 'frozen', False):
        # prefix with the helper name
        return exe, [exe, script_basename, *params]
    else:
        base   = os.path.dirname(os.path.abspath(__file__))
        helper = os.path.join(base, f"{script_basename}.py")
        return exe, [exe, "-u", helper, *params]

CREATE_NO_WINDOW = 0x08000000
CONFIG_FILE = "config.json"
BASE_WIDTH, BASE_HEIGHT = 1280, 1000

class TranslatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("EPUB Translator")
        master.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        master.minsize(1280, 1000)
        master.bind('<F11>', self.toggle_fullscreen)
        master.bind('<Escape>', lambda e: master.attributes('-fullscreen', False))
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")        
        # Warn on close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load config
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except:
            self.config = {}

        # Default prompts
        self.default_prompts = {
            "korean": "You are a professional Korean to English novel translator.\n- Use a context rich and natural translation style.\n- Retain honorifics, and suffixes like -nim, -ssi.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.",
            "japanese": "You are a professional Japanese to English novel translator.\n- Use a context rich and natural translation style.\n- Retain honorifics, and suffixes like -san, -sama, -chan, -kun.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.",
            "chinese": "You are a professional Chinese to English novel translator.\n- Use a context rich and natural translation style.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji."
        }

        # Profiles
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        self.profile_var = tk.StringVar(value=active)
        self.lang_var = self.profile_var

        # Main frame
        self.frame = tb.Frame(master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Grid config
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)
        self.frame.grid_columnconfigure(3, weight=1)
        self.frame.grid_columnconfigure(4, weight=0)
        for r in range(12):
            self.frame.grid_rowconfigure(r, weight=0)
        self.frame.grid_rowconfigure(9, weight=1, minsize=200)
        self.frame.grid_rowconfigure(10, weight=1, minsize=150)

        # EPUB File
        tb.Label(self.frame, text="EPUB File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_epub = tb.Entry(self.frame, width=50)
        self.entry_epub.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        tb.Button(self.frame, text="Browse", command=self.browse_file, width=12).grid(row=0, column=4, sticky=tk.EW, padx=5, pady=5)

        # Model
        tb.Label(self.frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value=self.config.get('model','gpt-4.1-nano'))
        tb.Combobox(self.frame, textvariable=self.model_var,
                    values=["gpt-4.1-nano","gpt-4.1-mini","gpt-4.1","gpt-3.5-turbo","gemini-1.5-pro","gemini-1.5-flash", "gemini-2.0-flash"], state="normal").grid(
            row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        # Language
        tb.Label(self.frame, text="Language:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.profile_menu = tb.Combobox(self.frame, textvariable=self.profile_var,
                                        values=list(self.prompt_profiles.keys()), state="normal")
        self.profile_menu.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.profile_menu.bind("<<ComboboxSelected>>", self.on_profile_select)
        self.profile_menu.bind("<Return>", self.on_profile_select)
        tb.Button(self.frame, text="Save Language", command=self.save_profile,
                  width=14).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        tb.Button(self.frame, text="Delete Language", command=self.delete_profile,
                  width=14).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        # Contextual
        self.contextual_var = tk.BooleanVar(value=self.config.get('contextual',True))
        tb.Checkbutton(self.frame, text="Contextual Translation",
                       variable=self.contextual_var).grid(row=3, column=0, columnspan=2,
                                                          sticky=tk.W, padx=5, pady=5)

        # API delay
        tb.Label(self.frame, text="API call delay (s):").grid(row=4, column=0,
                                                              sticky=tk.W, padx=5, pady=5)
        self.delay_entry = tb.Entry(self.frame, width=8)
        self.delay_entry.insert(0,str(self.config.get('delay',2)))
        self.delay_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ── New Chapter Range field ──
        tb.Label(self.frame, text="Chapter range:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.chapter_range_entry = tb.Entry(self.frame, width=12)
        # default could be “” or something like “1-5”
        self.chapter_range_entry.insert(0, self.config.get('chapter_range', ''))
        self.chapter_range_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        #Token limit
        tb.Label(self.frame, text="Token limit:").grid(row=6, column=0,sticky=tk.W, padx=5, pady=5)
        self.token_limit_entry = tb.Entry(self.frame, width=8)
        self.token_limit_entry.insert(0, str(self.config.get('token_limit', 1000000)))
        self.token_limit_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # ── Disable Token Limit button, placed below the token-limit field ──
        self.toggle_token_btn = tb.Button(
            self.frame,
            text="Disable Token Limit",
            command=self.toggle_token_limit,
            bootstyle="danger-outline",
            width=16
        )
        self.toggle_token_btn.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)

        # Translation settings
        tb.Label(self.frame, text="Temperature:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_temp = tb.Entry(self.frame, width=6)
        self.trans_temp.insert(0,str(self.config.get('translation_temperature',0.3)))
        self.trans_temp.grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)
        tb.Label(self.frame, text="Transl. Hist. Limit:").grid(row=5, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_history = tb.Entry(self.frame, width=6)
        self.trans_history.insert(0,str(self.config.get('translation_history_limit',3)))
        self.trans_history.grid(row=5, column=3, sticky=tk.W, padx=5, pady=5)

        # Glossary
        tb.Label(self.frame, text="Glossary Temp:").grid(row=6, column=2, sticky=tk.W, padx=5, pady=5)
        self.glossary_temp = tb.Entry(self.frame, width=6)
        self.glossary_temp.insert(0,str(self.config.get('glossary_temperature',0.3)))
        self.glossary_temp.grid(row=6, column=3, sticky=tk.W, padx=5, pady=5)
        tb.Label(self.frame, text="Glossary Hist. Limit:").grid(row=7, column=2, sticky=tk.W, padx=5, pady=5)
        self.glossary_history = tb.Entry(self.frame, width=6)
        self.glossary_history.insert(0,str(self.config.get('glossary_history_limit',3)))
        self.glossary_history.grid(row=7, column=3, sticky=tk.W, padx=5, pady=5)
        
                # ─── New GUI controls ───
        self.title_trim = tb.Entry(self.frame, width=6)
        self.title_trim.insert(0, str(self.config.get('title_trim_count', 1)))

        self.group_trim = tb.Entry(self.frame, width=6)
        self.group_trim.insert(0, str(self.config.get('group_affiliation_trim_count', 1)))

        self.traits_trim = tb.Entry(self.frame, width=6)
        self.traits_trim.insert(0, str(self.config.get('traits_trim_count', 1)))

        self.refer_trim = tb.Entry(self.frame, width=6)
        self.refer_trim.insert(0, str(self.config.get('refer_trim_count', 1)))

        self.loc_trim = tb.Entry(self.frame, width=6)
        self.loc_trim.insert(0, str(self.config.get('locations_trim_count', 1)))

        # API Key
        tb.Label(self.frame, text="OpenAI / Gemini API Key:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_key_entry = tb.Entry(self.frame, show='*')
        self.api_key_entry.insert(0,self.config.get('api_key',''))
        self.api_key_entry.grid(row=8, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        tb.Button(self.frame, text="Show", command=self.toggle_api_visibility,
                  width=12).grid(row=8, column=4, sticky=tk.EW, padx=5, pady=5)

        # System Prompt
        tb.Label(self.frame, text="System Prompt:").grid(row=9, column=0, sticky=tk.NW, padx=5, pady=5)
        self.prompt_text = tk.Text(
        self.frame,
        height=5,
        width=60,
        wrap='word',
        undo=True,              # turn on undo
        autoseparators=True,    # auto group edits
        maxundo=-1              # no fixed undo history limit
        )
        # bind keys
        self.prompt_text.bind('<Control-z>', lambda e: self.prompt_text.edit_undo())
        self.prompt_text.bind('<Control-y>', lambda e: self.prompt_text.edit_redo())
        self.prompt_text.grid(row=9, column=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)

        # Run Translation
        self.run_button = tb.Button(self.frame, text="Run Translation",
                                    command=self.run_translation_thread,
                                    bootstyle="success", width=14)
        self.run_button.grid(row=9, column=4, sticky=tk.N+tk.S+tk.EW, padx=5, pady=5)
        master.update_idletasks()
        self.run_base_w = self.run_button.winfo_width()
        self.run_base_h = self.run_button.winfo_height()
        master.bind('<Configure>', self.on_resize)

        # Log area
        self.log_text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD,
                                                  state=tk.DISABLED)
        self.log_text.grid(row=10, column=0, columnspan=5, sticky=tk.NSEW, padx=5, pady=5)

        # Bottom toolbar
        self._make_bottom_toolbar()
        
        self.token_limit_disabled = False

        # initial prompt
        self.on_profile_select()

        
    def on_close(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit? Unsaved translations will be stopped."):
            # Terminate any running subprocesses
            for proc_attr in ('proc', 'fallback_proc', 'glossary_proc'):
                proc = getattr(self, proc_attr, None)
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            self.master.destroy()
            # Ensure the interpreter exits
            self.root.destroy()
            sys.exit(0)

    def toggle_token_limit(self):
        """Toggle whether the token-limit entry is active or not."""
        if not self.token_limit_disabled:
            # disable it
            self.token_limit_entry.delete(0, tk.END)
            self.token_limit_entry.config(state=tk.DISABLED)
            self.toggle_token_btn.config(text="Enable Token Limit", bootstyle="success-outline")
            self.append_log("⚠️ Token limit disabled.")
        else:
            # re-enable it
            self.token_limit_entry.config(state=tk.NORMAL)
            # restore a default or previous value—here we use 1,000,000
            self.token_limit_entry.insert(0, str(self.config.get('token_limit', 1000000)))
            self.toggle_token_btn.config(text="Disable Token Limit", bootstyle="danger-outline")
            self.append_log("✅ Token limit re-enabled.")
        # flip the flag
        self.token_limit_disabled = not self.token_limit_disabled
        
    def _make_bottom_toolbar(self):
        # 1) toolbar on row 11
        btn_frame = tb.Frame(self.frame)
        btn_frame.grid(row=11, column=0, columnspan=5, sticky=tk.EW, pady=5)

        toolbar_items = [
            ("EPUB Converter",      self.epub_fallback,               "info"),
            ("Extract Glossary",    self.run_glossary_extraction_thread, "warning"),
            ("Trim Glossary",       self.trim_glossary,               "secondary"),
            ("Save Config",         self.save_config,                 "secondary"),
            ("Load Glossary",       self.load_glossary,               "secondary"),
            ("Import Profiles",     self.import_profiles,             "secondary"),
            ("Export Profiles",     self.export_profiles,             "secondary"),
        ]
        for idx, (lbl, cmd, style) in enumerate(toolbar_items):
            btn_frame.columnconfigure(idx, weight=1)
            btn = tb.Button(btn_frame, text=lbl, command=cmd, bootstyle=style)
            btn.grid(row=0, column=idx, sticky=tk.EW, padx=2)
            if lbl == "Extract Glossary":
                self.glossary_button = btn

        # 2) make sure row 12 exists
        self.frame.grid_rowconfigure(12, weight=0)



    def trim_glossary(self):
        # 1) Load the list once, so we can pass it to the dialog
        path = filedialog.askopenfilename(
            title="Select glossary.json to trim",
            filetypes=[("JSON files","*.json")]
        )
        if not path:
            return

        with open(path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)

        # 2) Build a dialog for all six controls
        dlg = tk.Toplevel(self.master)
        dlg.title("Glossary Trimmer")
        dlg.geometry("420x425")
        dlg.transient(self.master)   # keep on top
        dlg.grab_set()               # modal

        labels = [
            "Entries (appearance order):",             # how many entries to keep
            "Traits Trim Count:",        # drop last M traits (0=remove all)
            "Title Keep (0=remove):",    # 0 to drop title field
            "GroupAffil Trim Count:",    # drop last M affiliations
            "Ref-To-Others Trim Count:", # drop last M name-map entries
            "Locations Trim Count:"      # drop last M locations
        ]
        # default values from current GUI sliders
        defaults = [
            "100",
            self.traits_trim.get(),
            self.title_trim.get(),
            self.group_trim.get(),
            self.refer_trim.get(),
            self.loc_trim.get()
        ]
        entries = []
        for i,(lab,defval) in enumerate(zip(labels,defaults)):
            tb.Label(dlg, text=lab).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            e = tb.Entry(dlg, width=6)
            e.insert(0, defval)
            e.grid(row=i, column=1, padx=5, pady=2)
            entries.append(e)

        # 3) When user clicks Apply, run the same trimming logic
        
        def aggregate_locations():
            all_locs = []
            for char in glossary:
                locs = char.get('locations', [])
                if isinstance(locs, list):
                    all_locs.extend(locs)
                # Remove location entry from character
                char.pop('locations', None)

            # Deduplicate while preserving order
            seen = set()
            unique_locs = []
            for loc in all_locs:
                if loc not in seen:
                    seen.add(loc)
                    unique_locs.append(loc)

            # Optional: remove old "Location Summary" if already present
            glossary[:] = [entry for entry in glossary if entry.get('original_name') != "📍 Location Summary"]

            # Append the new summary
            glossary.append({
                "original_name": "📍 Location Summary",
                "name": "Location Summary",
                "locations": unique_locs
            })

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("Aggregated", f"{len(unique_locs)} unique locations added to glossary.")
            dlg.lift()

        
        def apply_trim():
            top_limit, traits_lim, title_lim, group_lim, refer_lim, loc_lim = (
                int(e.get()) for e in entries
            )
            # top‐level slice
            trimmed = glossary[:top_limit]
            # per‐entry pruning (same as before)…
            for char in trimmed:
                if title_lim <= 0:
                    char.pop('title', None)
                if traits_lim <= 0:
                    char.pop('traits', None)
                else:
                    t = char.get('traits', [])
                    char['traits'] = t[:-traits_lim] if len(t)>traits_lim else []
                if group_lim <= 0:
                    char.pop('group_affiliation', None)
                else:
                    g = char.get('group_affiliation', [])
                    char['group_affiliation'] = g[:-group_lim] if len(g)>group_lim else []
                if refer_lim <= 0:
                    char.pop('how_they_refer_to_others', None)
                else:
                    items = list(char.get('how_they_refer_to_others',{}).items())
                    keep = items[:-refer_lim] if len(items)>refer_lim else []
                    char['how_they_refer_to_others'] = dict(keep)
                if loc_lim <= 0:
                    char.pop('locations', None)
                else:
                    l = char.get('locations', [])
                    char['locations'] = l[:-loc_lim] if len(l)>loc_lim else []

            # overwrite file
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("Trimmed", f"Glossary written with {top_limit} entries.")
            dlg.destroy()

        tb.Button(dlg, text="Apply", command=apply_trim, bootstyle="success") \
          .grid(row=len(labels), column=0, columnspan=2, pady=10)
        tb.Button(dlg, text="➕ Aggregate Unique Locations",
                  command=aggregate_locations, bootstyle="info") \
          .grid(row=len(labels)+1, column=0, columnspan=2, pady=5)         
            # 1) define the helper
        def delete_empty_fields():
            for char in glossary:
                for key in list(char.keys()):
                    val = char[key]
                    if val in (None, [], {}, ""):
                        char.pop(key, None)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Deleted", "Empty fields removed.")
            dlg.lift()

        # 2) place the button (note the commas after each kwarg!)
        tb.Button(
            dlg,
            text="Delete Empty Fields",          # ← comma here
            command=delete_empty_fields,         # ← *and* here
            bootstyle="warning"                  # no trailing comma needed but allowed
        ).grid(
            row=len(labels)+2,
            column=0,
            columnspan=2,
            pady=5
        )

        dlg.wait_window()



    def on_resize(self, event):
        if event.widget is self.master:
            sx = event.width / BASE_WIDTH
            sy = event.height / BASE_HEIGHT
            s = min(sx, sy)
            new_w = int(self.run_base_w * s)
            new_h = int(self.run_base_h * s)
            ipadx = max(0, (new_w - self.run_base_w)//2)
            ipady = max(0, (new_h - self.run_base_h)//2)
            self.run_button.grid_configure(ipadx=ipadx, ipady=ipady)
        
      
    def save_profiles(self):
        """Persist only the prompt profiles and active profile."""
        try:
            data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            data['prompt_profiles'] = self.prompt_profiles
            data['active_profile'] = self.profile_var.get()
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save profiles: {e}")

    def save_profile(self):
        """Save current prompt under selected profile and persist."""
        name = self.profile_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Language name cannot be empty.")
            return
        content = self.prompt_text.get('1.0', tk.END).strip()
        self.prompt_profiles[name] = content
        self.config['prompt_profiles'] = self.prompt_profiles
        self.config['active_profile'] = name
        self.profile_menu['values'] = list(self.prompt_profiles.keys())
        messagebox.showinfo("Saved", f"Language '{name}' saved.")
        self.save_profiles()

    def delete_profile(self):
        """Delete selected profile and persist."""
        name = self.profile_var.get()
        if name not in self.prompt_profiles:
            messagebox.showerror("Error", f"Language '{name}' not found.")
            return
        if messagebox.askyesno("Delete", f"Delete language '{name}'?"):
            del self.prompt_profiles[name]
            self.config['prompt_profiles'] = self.prompt_profiles
            if self.prompt_profiles:
                new = next(iter(self.prompt_profiles))
                self.profile_var.set(new)
                self.on_profile_select()
            else:
                self.profile_var.set("")
                self.prompt_text.delete('1.0', tk.END)
            self.profile_menu['values'] = list(self.prompt_profiles.keys())
            self.save_profiles()


    def delete_profile(self):
        """Delete the selected language/profile."""
        name = self.profile_var.get()
        if name not in self.prompt_profiles:
            messagebox.showerror("Error", f"Language '{name}' not found.")
            return
        if messagebox.askyesno("Delete", f"Are you sure you want to delete language '{name}'?"):
            del self.prompt_profiles[name]
            self.config['prompt_profiles'] = self.prompt_profiles
            if self.prompt_profiles:
                new = next(iter(self.prompt_profiles))
                self.profile_var.set(new)
                self.on_profile_select()
            else:
                self.profile_var.set("")
                self.prompt_text.delete('1.0', tk.END)
            self.profile_menu['values'] = list(self.prompt_profiles.keys())
        self.save_profiles()

    def on_profile_select(self, event=None):
        """Load the selected profile's prompt into the text area."""
        name = self.profile_var.get()
        prompt = self.prompt_profiles.get(name, "")
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", prompt)
        
    def import_profiles(self):
        """Import profiles from a JSON file, merging into existing ones."""
        path = filedialog.askopenfilename(title="Import Profiles", filetypes=[("JSON files","*.json")])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.prompt_profiles.update(data)
            self.config['prompt_profiles'] = self.prompt_profiles
            self.profile_menu['values'] = list(self.prompt_profiles.keys())
            messagebox.showinfo("Imported", f"Imported {len(data)} profiles.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import profiles: {e}")

    def export_profiles(self):
        """Export all profiles to a JSON file."""
        path = filedialog.asksaveasfilename(title="Export Profiles", defaultextension=".json", filetypes=[("JSON files","*.json")])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_profiles, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Exported", f"Profiles exported to {path}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export profiles: {e}")
            
    def set_default_prompt(self):
        prompt = self.default_prompts.get(self.lang_var.get(), "")
        self.prompt_text.delete('1.0', tk.END)
        self.prompt_text.insert('1.0', prompt)

    def load_glossary(self):
        """Let the user pick a glossary.json and remember its path."""
        path = filedialog.askopenfilename(
            title="Select glossary.json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        # store it for later
        self.manual_glossary_path = path
        self.log_debug(f"📑 Loaded manual glossary: {path}")

    def toggle_fullscreen(self, event=None):
        is_full = self.master.attributes('-fullscreen')
        self.master.attributes('-fullscreen', not is_full)

    def toggle_api_visibility(self):
        show = self.api_key_entry.cget('show')
        self.api_key_entry.config(show='' if show == '*' else '*')

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("EPUB files","*.epub")])
        if path:
            self.entry_epub.delete(0, tk.END)
            self.entry_epub.insert(0, path)

    def append_log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def run_translation_thread(self):
        self.run_button_state(False)
        threading.Thread(target=self.run_translation, daemon=True).start()

    def run_button_state(self, enabled):
        # Find run button and enable/disable
        for child in self.frame.winfo_children():
            if isinstance(child, tb.Button) and child.cget('text') == 'Run Translation':
                child.config(state=tk.NORMAL if enabled else tk.DISABLED)
                break

    def run_translation(self):
        epub_path = self.entry_epub.get()
        epub_path   = self.entry_epub.get()
        epub_base   = os.path.splitext(os.path.basename(epub_path))[0]
        output_dir  = os.path.join(os.getcwd(), epub_base)
        history_file = os.path.join(output_dir, "translation_history.json")
        api_key   = self.api_key_entry.get()
        delay     = self.delay_entry.get()
        model     = self.model_var.get()
        lang      = self.lang_var.get().lower()
        contextual= self.contextual_var.get()
        sys_prompt= self.prompt_text.get("1.0", "end").strip()

        # --- validation ---
        if not epub_path or not os.path.isfile(epub_path):
            messagebox.showerror("Error", "Please select a valid EPUB file.")
            return self._reenable()
        if not api_key:
            messagebox.showerror("Error", "Please enter your OpenAI or Gemini API key.")
            return self._reenable()
        try:
            delay = int(delay)
        except ValueError:
            messagebox.showerror("Error", "Delay must be an integer.")
            return self._reenable()

        # --- save config ---
        cfg = {
            "api_key": api_key,
            "delay": delay,
            "model": model,
            "lang": lang,
            "contextual": contextual,
            "system_prompt": sys_prompt,
            "translation_temperature": float(self.trans_temp.get()),
            "translation_history_limit":  int(self.trans_history.get())
        }
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        # --- build env & launch ---
        env = os.environ.copy()
        env["TRANSLATION_HISTORY_LIMIT"] = str(self.trans_history.get())
        env['TRANSLATION_TEMPERATURE'] = str(self.trans_temp.get())
        # point history I/O at the global Payloads folder:
        env["EPUB_OUTPUT_DIR"] = os.getcwd()       
        env['EPUB_PATH'] = epub_path
        env['MODEL'] = model
        env['CONTEXTUAL'] = '1' if contextual else '0'
        env['SEND_INTERVAL_SECONDS'] = str(delay)
        env['OPENAI_API_KEY'] = api_key
        env['SYSTEM_PROMPT']    = self.prompt_text.get("1.0", "end").strip()
        # new:
        chap_range = self.chapter_range_entry.get().strip()
        if chap_range:
            env['CHAPTER_RANGE'] = chap_range
        token_val = self.token_limit_entry.get().strip()
        if token_val:
            env['TOKEN_LIMIT'] = token_val
        else:
            env.pop('TOKEN_LIMIT', None)
        # ← insert here ↓
        if hasattr(self, 'manual_glossary_path'):
            env['MANUAL_GLOSSARY'] = self.manual_glossary_path
            self.log_debug(f"  MANUAL_GLOSSARY = {self.manual_glossary_path}")
        env['TRANSLATION_LANG'] = self.lang_var.get().lower()

        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        
        self.log_debug("📦 Environment Variables Set:")
        self.log_debug(f"  EPUB_PATH = {epub_path}")
        self.log_debug(f"  MODEL = {model}")
        self.log_debug(f"  CONTEXTUAL = {contextual}")
        self.log_debug(f"  DELAY = {delay}")
        self.log_debug(f"  TRANSLATION_TEMPERATURE = {self.trans_temp.get()}")
        self.log_debug(f"  TRANSLATION_HISTORY_LIMIT = {self.trans_history.get()}")
        self.log_debug(f"  API_KEY = {'*' * len(api_key) if api_key else '(not provided)'}")
        self.log_debug("  SYSTEM_PROMPT:")
        # …then each line of the prompt indented one level further
        for line in env['SYSTEM_PROMPT'].splitlines():
            self.log_debug(f"    {line}")
        # blank separator
        self.log_debug("")
        # follow with the next setting
        self.log_debug(f"  TRANSLATION_LANG = {env['TRANSLATION_LANG']}")
        self.log_debug("🚀 Launching translation subprocess...\n")
        self.append_log("🚀 Starting translation subprocess…")
        
        #  Launch translation subprocess
        exe, args = build_subprocess_args('TransateKRtoEN', epub_path)
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding='utf-8',
            errors='ignore',
            creationflags=CREATE_NO_WINDOW,
            env=env,
        )

        # ▶️ Stream its output
        for line in self.proc.stdout:
            self.append_log(line.rstrip())

        # ▶️ Wait for it to finish
        self.proc.wait()

        # ▶️ Check exit code
        if self.proc.returncode == 0:
            messagebox.showinfo("Success", "Translation complete!")
        else:
            self.append_log(f"❌ Translation failed (code {self.proc.returncode}).")
            if messagebox.askyesno("Run Fallback?",
                                   "Translation failed. Run EPUB fallback compiler?"):
                exe_fb, args_fb = build_subprocess_args(
                    'epub_fallback_compiler_with_cover_portable',
                    output_dir
                )
                fb = subprocess.Popen(
                    args_fb,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in fb.stdout:
                    self.append_log("[FALLBACK] " + line.rstrip())
                fb.wait()
                if fb.returncode == 0:
                    messagebox.showinfo("Fallback Success",
                                        "Fallback EPUB compiled successfully.")
                else:
                    messagebox.showerror("Fallback Failed",
                                         f"Exit code {fb.returncode}")

        # ▶️ Re-enable UI
        self._reenable()

    def _reenable(self):
        self.run_button.config(state=tk.NORMAL)
        
    def run_glossary_extraction_thread(self):
        """Spawn a thread so the GUI doesn’t freeze."""
        self.glossary_button.config(state=tk.DISABLED)
        threading.Thread(target=self.run_glossary_extraction, daemon=True).start()

    def run_glossary_extraction(self):
        epub_path = self.entry_epub.get()
        temp = self.glossary_temp.get()
        history = self.glossary_history.get()



        # Validate EPUB path
        if not epub_path or not os.path.isfile(epub_path):
            messagebox.showerror("Error", "Please select a valid EPUB file for glossary extraction.")
            self.glossary_button.config(state=tk.NORMAL)
            return

        # Validate entries
        try:
            temp_val = float(temp)
            hist_val = int(history)
        except ValueError:
            messagebox.showerror("Error", "Temp must be a float and Glossary History Limit an integer.")
            self.glossary_button.config(state=tk.NORMAL)
            return

        # Persist settings
        self.config['temperature'] = temp_val
        self.config['glossary_history_limit'] = hist_val
        self.config['title_trim_count']            = int(self.title_trim.get())
        self.config['group_affiliation_trim_count'] = int(self.group_trim.get())
        self.config['traits_trim_count']      = int(self.traits_trim.get())
        self.config['refer_trim_count']            = int(self.refer_trim.get())
        self.config['locations_trim_count']        = int(self.loc_trim.get())
        

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        # Build subprocess command
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        script = os.path.join(base_dir, "extract_glossary_from_epub.py")

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['GLOSSARY_TEMPERATURE'] = temp
        env['GLOSSARY_CONTEXT_LIMIT'] = history
        
        # --- Debug logging ---
        self.log_debug("🚀 Starting glossary extraction subprocess…")
        self.log_debug(f"  EPUB_PATH = {epub_path}")
        self.log_debug(f"  GLOSSARY_TEMPERATURE = {temp}")
        self.log_debug(f"  GLOSSARY_CONTEXT_LIMIT = {history}")
        self.log_debug("📄 Extracting chapters for glossary…")
        self.log_debug("")

        # ▶️ Launch glossary extraction subprocess
        exe, args = build_subprocess_args(
            'extract_glossary_from_epub',
            '--epub', epub_path,
            '--config', CONFIG_FILE
        )
        self.glossary_proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )

        # ▶️ Stream its output
        for line in self.glossary_proc.stdout:
            self.append_log(line.rstrip())

        # ▶️ Wait for it to finish
        self.glossary_proc.wait()

        # ▶️ Check exit code
        if self.glossary_proc.returncode == 0:
            messagebox.showinfo("Glossary", "Extraction finished!")
        else:
            messagebox.showerror(
                "Glossary Error",
                f"Exit code {self.glossary_proc.returncode}"
            )

        # ▶️ Re-enable the button
        self.glossary_button.config(state=tk.NORMAL)

    def epub_fallback(self):
        
        folder = filedialog.askdirectory(title="Select translation output folder")
        if not folder:
            return
        
        self.append_log("📦 [DEBUG] Running EPUB fallback...")
        try:

                        # build args for fallback compiler
            exe, args = build_subprocess_args(
                'epub_fallback_compiler_with_cover_portable',
                folder
            )
            self.fallback_proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in self.fallback_proc.stdout:
                self.append_log(line.rstrip())
            self.fallback_proc.wait()

            if self.fallback_proc.returncode == 0:
                out_file = os.path.join(folder, "translated_fallback.epub")
                self.append_log(f"✅ Fallback created: {out_file}")
                messagebox.showinfo("Fallback Success", f"Created: {out_file}")
            else:
                self.append_log(f"❌ Fallback failed (code {self.fallback_proc.returncode})")
                messagebox.showerror("Fallback Failed", f"Exit code {self.fallback_proc.returncode}")

        except Exception as e:
            self.append_log(f"❌ Could not launch fallback: {e}")

    def save_config(self):
        """Persist all settings to config.json without referencing lang_var."""
        try:
            # Collect settings
            self.config['model'] = self.model_var.get()
            self.config['active_profile'] = self.profile_var.get()
            self.config['prompt_profiles'] = self.prompt_profiles
            self.config['contextual'] = self.contextual_var.get()
            self.config['delay'] = int(self.delay_entry.get())
            self.config['translation_temperature'] = float(self.trans_temp.get())
            self.config['translation_history_limit'] = int(self.trans_history.get())
            self.config['glossary_temperature'] = float(self.glossary_temp.get())
            self.config['glossary_history_limit'] = int(self.glossary_history.get())
            self.config['api_key'] = self.api_key_entry.get()
            # defensively handle empty/disabled token‐limit field
            _tl = self.token_limit_entry.get().strip()
            if _tl.isdigit():
                self.config['token_limit'] = int(_tl)
            else:
                self.config['token_limit'] = None

            # Write to file
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", "Configuration saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
        
    def log_debug(self, message):
        # Temporarily turn the text widget back to NORMAL so we can insert…
        self.log_text.configure(state=tk.NORMAL)
        # …then log the line with a [DEBUG] prefix
        self.log_text.insert(tk.END, f"[DEBUG] {message}\n")
        self.log_text.see(tk.END)
        # And immediately disable again so the user can’t edit it
        self.log_text.configure(state=tk.DISABLED)


if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = TranslatorGUI(root)
    root.mainloop()
