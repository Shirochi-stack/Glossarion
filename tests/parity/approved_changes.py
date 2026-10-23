"""Owner-approved desktop default changes made after the goldens and frozen literals were captured.

The parity tests replay the desktop as it was at their pinned commits (``git show <sha>:...``)
and compare it with recorded goldens. A deliberate default change (DISCREPANCIES.md: "each
needs a separate, user-approved change that also updates the goldens") is applied here, to the
legacy source the tests read and to the goldens they load, so every other behaviour is still
checked exactly. Each entry names the source rows and the recorded values it changes.
"""
from __future__ import annotations


def _after(line, *added):
    """Insert *added* lines after *line* (both without their newline)."""
    return (line + "\n", line + "\n" + "".join(text + "\n" for text in added))


# 2026-10-17 "Always check and respect Retry-After" (respect_retry_after / RESPECT_RETRY_AFTER)
_RESPECT_SETTINGS_ROW = _after(
    "            ('indefinite_rate_limit_retry', ['indefinite_rate_limit_retry_var'], False, bool),",
    "            ('respect_retry_after', ['respect_retry_after_var'], True, bool),")
_RESPECT_SETTINGS_ROW_GUI = _after(
    "                ('indefinite_rate_limit_retry', ['indefinite_rate_limit_retry_var'], False, bool),",
    "                ('respect_retry_after', ['respect_retry_after_var'], True, bool),")
_RESPECT_BOOL_ROW = _after(
    "            ('indefinite_rate_limit_retry_var', 'indefinite_rate_limit_retry', False),",
    "            ('respect_retry_after_var', 'respect_retry_after', True),")
_RESPECT_MIGRATION = _after(
    "            setattr(self, var_name, create_var(bool, key, default))\n"
    "        self.use_header_as_output_var = False",
    "        if self.respect_retry_after_var and getattr(self, 'ignore_retry_after_var', False):",
    "            # 'Always check and respect Retry-After' contradicts 'Ignore server Retry-After header':",
    "            # a config holding both keeps the explicit ignore choice.",
    "            self.respect_retry_after_var = False")
_RESPECT_ENV_DICT = _after(
    "            'INDEFINITE_RATE_LIMIT_RETRY': '1' if self.config.get('indefinite_rate_limit_retry', False) else '0',",
    "            'RESPECT_RETRY_AFTER': '1' if getattr(self, 'respect_retry_after_var', True) else '0',")
_RESPECT_ENV_PAIR = _after(
    "                ('INDEFINITE_RATE_LIMIT_RETRY', '1' if getattr(self, 'indefinite_rate_limit_retry_var', False) else '0'),",
    "                ('RESPECT_RETRY_AFTER', '1' if getattr(self, 'respect_retry_after_var', True) else '0'),")

# 2026-10-18 per-category Gemini safety thresholds (gemini_safety_category_thresholds /
# GEMINI_SAFETY_CATEGORY_THRESHOLDS)
_SAFETY_CATEGORY_ROW = _after(
    "            ('gemini_safety_threshold', ['gemini_safety_threshold_var'], 'BLOCK_NONE', str),",
    "            ('gemini_safety_category_thresholds', ['gemini_safety_category_thresholds_var'], {}, dict),")
_SAFETY_CATEGORY_ROW_GUI = _after(
    "                ('gemini_safety_threshold', ['gemini_safety_threshold_var'], 'BLOCK_NONE', str),",
    "                ('gemini_safety_category_thresholds', ['gemini_safety_category_thresholds_var'], {}, dict),")
_SAFETY_CATEGORY_ENV_DICT = _after(
    "            'GEMINI_SAFETY_THRESHOLD': str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE')),",
    "            'GEMINI_SAFETY_CATEGORY_THRESHOLDS': json.dumps(self.config.get('gemini_safety_category_thresholds') or {}),")
_SAFETY_CATEGORY_ENV_PAIR = _after(
    "                ('GEMINI_SAFETY_THRESHOLD', str(self.config.get('gemini_safety_threshold', 'BLOCK_NONE'))),",
    "                ('GEMINI_SAFETY_CATEGORY_THRESHOLDS', json.dumps(self.config.get('gemini_safety_category_thresholds') or {})),")

# 2026-10-18 built-in On-device translation prompt profiles (small on-device models); 2026-10-19 split by
# text extraction method: On-device_BeautifulSoup (html tags, the default) and On-device_html2text (markdown)
_ON_DEVICE_PROMPT = "You must only output {target_lang} translations while retaining the html tags of the following text."
_ON_DEVICE_HTML2TEXT_PROMPT = "You must only output {target_lang} translations with markdown."
# one row patches both identical lists (always_include_profiles and the protected set)
_ON_DEVICE_PROFILE_LIST_ROW = _after('            "Subtitle Translation",', '            "On-device_BeautifulSoup",',
                                     '            "On-device_html2text",')
_ON_DEVICE_DEFAULT_PROMPT_ROW = _after(
    '            "Subtitle Translation": DEFAULT_SUBTITLE_TRANSLATION_PROMPT,',
    f'            "On-device_BeautifulSoup": "{_ON_DEVICE_PROMPT}",',
    f'            "On-device_html2text": "{_ON_DEVICE_HTML2TEXT_PROMPT}",')

# 2026-10-18 {split_marker_instruction} stays where the user put it: filled in place, or its own
# line removed without joining the lines around it (prompt_defaults.apply_split_marker_instruction)
_SPLIT_MARKER_OLD_SUB = ("re.sub(r'\\s*\\{split_marker_instruction\\}\\s*', "
                         "lambda m: '\\n' + split_instr if split_instr else '', system_prompt)")


def _split_marker_fill(indent):
    return (indent + "# Use a regex to replace the placeholder regardless of surrounding whitespace or newlines\n"
            + indent + "import re\n"
            + indent + "system_prompt = " + _SPLIT_MARKER_OLD_SUB + "\n",
            indent + "# Fill the placeholder where the user put it; without an instruction its line goes away\n"
            + indent + "from prompt_defaults import apply_split_marker_instruction\n"
            + indent + "system_prompt = apply_split_marker_instruction(system_prompt, split_instr)\n")


_SPLIT_MARKER_TEXT_FILE = _split_marker_fill(" " * 16)   # _process_text_file (text_jobs)
_SPLIT_MARKER_IMAGE_FILE = _split_marker_fill(" " * 12)  # _process_image_file (image_job)
_SPLIT_MARKER_RPGMAKER = (                               # _process_rpgmaker_game (rpgmaker_job)
    "            import re\n"
    "            system_prompt = re.sub(r'\\s*\\{split_marker_instruction\\}\\s*', '', system_prompt)\n",
    "            from prompt_defaults import apply_split_marker_instruction\n"
    "            system_prompt = apply_split_marker_instruction(system_prompt, \"\")\n")

# 2026-10-18 Gemini safety: each harm category's own threshold (gemini_policy.gemini_safety_thresholds)
# plus HARM_CATEGORY_JAILBREAK, also in the async batch request builder (it sent BLOCK_NONE for five).
_BATCH_SAFETY = (
    '            if os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true":\n'
    '                request["generateContentRequest"]["safetySettings"] = [\n'
    '                    {"category": cat, "threshold": "BLOCK_NONE"}\n'
    '                    for cat in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",\n'
    '                               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",\n'
    '                               "HARM_CATEGORY_CIVIC_INTEGRITY"]\n'
    '                ]\n',
    '            if os.getenv("DISABLE_GEMINI_SAFETY", "false").lower() == "true":\n'
    '                from gemini_policy import gemini_safety_thresholds\n'
    '                request["generateContentRequest"]["safetySettings"] = [\n'
    '                    {"category": cat, "threshold": threshold}\n'
    '                    for cat, threshold in gemini_safety_thresholds("BLOCK_NONE")\n'
    '                ]\n',
)

# 2026-10-19 "Skip title tag translation" is on by default (skip_title_tag_translation /
# SKIP_TITLE_TAG_TRANSLATION; the inverse legacy use_title / USE_TITLE follows)
_TITLE_TAG_CONFIG_GET = ("self.config.get('skip_title_tag_translation', False)",
                         "self.config.get('skip_title_tag_translation', True)")
_TITLE_TAG_GETATTR = ("getattr(self, 'skip_title_tag_translation_var', False)",
                      "getattr(self, 'skip_title_tag_translation_var', True)")
_TITLE_TAG_SETTINGS_ROW = ("('skip_title_tag_translation', ['skip_title_tag_translation_var'], False, bool)",
                           "('skip_title_tag_translation', ['skip_title_tag_translation_var'], True, bool)")
_USE_TITLE_SETTINGS_ROW = ("('use_title', ['use_title_var'], True, bool)",
                           "('use_title', ['use_title_var'], False, bool)")

# 2026-10-19 "Remove excessive empty p and br tags" (On-device: remove_excessive_empty_tags_on_device /
# REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE, on; all models: remove_excessive_empty_tags_all_models /
# REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS, off). Rows follow the (already approved) title tag rows.
_EMPTY_TAGS_SETTINGS_ROWS = (
    "('remove_excessive_empty_tags_on_device', ['remove_excessive_empty_tags_on_device_var'], True, bool),",
    "('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),",
)
_EMPTY_TAGS_SETTINGS_ROW = _after(
    "            ('skip_title_tag_translation', ['skip_title_tag_translation_var'], True, bool),",
    *("            " + row for row in _EMPTY_TAGS_SETTINGS_ROWS))
_EMPTY_TAGS_SETTINGS_ROW_GUI = _after(
    "                ('skip_title_tag_translation', ['skip_title_tag_translation_var'], True, bool),",
    *("                " + row for row in _EMPTY_TAGS_SETTINGS_ROWS))
_EMPTY_TAGS_BOOL_ROW = _after(
    "            ('respect_retry_after_var', 'respect_retry_after', True),",
    "            ('remove_excessive_empty_tags_on_device_var', 'remove_excessive_empty_tags_on_device', True),",
    "            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),")
_EMPTY_TAGS_ENV_DICT = _after(
    "            'RESPECT_RETRY_AFTER': '1' if getattr(self, 'respect_retry_after_var', True) else '0',",
    "            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, "
    "'remove_excessive_empty_tags_on_device_var', True) else '0',",
    "            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, "
    "'remove_excessive_empty_tags_all_models_var', False) else '0',")
_EMPTY_TAGS_ENV_PAIR = _after(
    "                ('RESPECT_RETRY_AFTER', '1' if getattr(self, 'respect_retry_after_var', True) else '0'),",
    "                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, "
    "'remove_excessive_empty_tags_on_device_var', True) else '0'),",
    "                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, "
    "'remove_excessive_empty_tags_all_models_var', False) else '0'),")
# the async batch dialog (frozen async_api_processor) exports the all-models toggle and applies it
_EMPTY_TAGS_ASYNC_ENV = _after(
    "        env_vars['REQUEST_MERGING_ENABLED'] = '0'",
    "        env_vars['REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS'] = \"1\" if _val(getattr(self.gui, "
    "'remove_excessive_empty_tags_all_models_var', None), self.gui.config.get("
    "'remove_excessive_empty_tags_all_models', False)) else \"0\"")
_EMPTY_TAGS_ASYNC_PAYLOAD = _after(
    "                                else:\n"
    "                                    chapter_payload = chapter_html",
    "                                    if env_vars.get('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS') == '1':",
    "                                        # \"Remove excessive empty p and br tags (all models)\": the text sent only",
    "                                        from html_duplicate_cleanup import collapse_empty_paragraphs",
    "                                        chapter_payload = collapse_empty_paragraphs(chapter_payload)")

# 2026-10-19 "⏳ Request Queue" of the main GUI key (request_queue_schedule / REQUEST_QUEUE_SCHEDULE, JSON;
# {} = off): requests wait before sending during its schedule windows. Rows follow the respect_retry_after rows.
_QUEUE_SETTINGS_ROW = _after(
    "            ('respect_retry_after', ['respect_retry_after_var'], True, bool),",
    "            ('request_queue_schedule', ['request_queue_schedule_var'], {}, dict),")
_QUEUE_SETTINGS_ROW_GUI = _after(
    "                ('respect_retry_after', ['respect_retry_after_var'], True, bool),",
    "                ('request_queue_schedule', ['request_queue_schedule_var'], {}, dict),")
_QUEUE_ENV_DICT = _after(
    "            'GEMINI_SAFETY_CATEGORY_THRESHOLDS': json.dumps(self.config.get('gemini_safety_category_thresholds') or {}),",
    "            'REQUEST_QUEUE_SCHEDULE': json.dumps(self.config.get('request_queue_schedule') or {}),")
_QUEUE_ENV_PAIR = _after(
    "                ('GEMINI_SAFETY_CATEGORY_THRESHOLDS', json.dumps(self.config.get('gemini_safety_category_thresholds') or {})),",
    "                ('REQUEST_QUEUE_SCHEDULE', json.dumps(self.config.get('request_queue_schedule') or {})),")

# 2026-10-21 RapidOCR reading model (rapidocr_language) on the desktop Manga tab: a combo under the
# OCR provider row, shown while RapidOCR is the provider (MangaTranslationTab._build_pyside6_interface /
# _on_ocr_provider_change)
_RAPIDOCR_FRAME = _after(
    "        self.custom_api_ocr_batch_frame.setVisible(self.ocr_provider_value == 'custom-api')",
    "",
    "        # RapidOCR reading model (rapidocr_language; the choices and help are the settings schema's,",
    "        # which the mobile Settings show too)",
    "        from settings_schema import CHOICES_OVERRIDES, TOOLTIP_OVERRIDES",
    "        self.rapidocr_frame = QWidget()",
    "        rapidocr_layout = QHBoxLayout(self.rapidocr_frame)",
    "        rapidocr_layout.setContentsMargins(0, 0, 0, 14)",
    "        rapidocr_layout.setSpacing(10)",
    "        rapidocr_label = QLabel(\"Reading model:\")",
    "        rapidocr_label.setFixedWidth(ocr_label_column_width)",
    "        rapidocr_label.setAlignment(Qt.AlignLeft)",
    "        rapidocr_layout.addWidget(rapidocr_label)",
    "        self.rapidocr_language_combo = QComboBox()",
    "        for value, label in CHOICES_OVERRIDES['rapidocr_language']:",
    "            self.rapidocr_language_combo.addItem(label, value)",
    "        current = self.rapidocr_language_combo.findData(getattr(self, 'rapidocr_language_value', 'auto'))",
    "        self.rapidocr_language_combo.setCurrentIndex(max(0, current))",
    "        self.rapidocr_language_combo.setToolTip(TOOLTIP_OVERRIDES['rapidocr_language'])",
    "        self.rapidocr_language_combo.currentIndexChanged.connect(",
    "            lambda _index: (",
    "                setattr(self, 'rapidocr_language_value', self.rapidocr_language_combo.currentData()),",
    "                self._save_rendering_settings()",
    "            )",
    "        )",
    "        self._disable_combobox_mousewheel(self.rapidocr_language_combo)",
    "        rapidocr_layout.addWidget(self.rapidocr_language_combo)",
    "        rapidocr_layout.addStretch()",
    "        settings_frame_layout.addWidget(self.rapidocr_frame)",
    "        self.rapidocr_frame.setVisible(self.ocr_provider_value == 'rapidocr')")
_RAPIDOCR_FRAME_VISIBILITY = _after(
    "            self.custom_api_ocr_batch_frame.setVisible(provider == 'custom-api')",
    "        if hasattr(self, 'rapidocr_frame'):",
    "            self.rapidocr_frame.setVisible(provider == 'rapidocr')")
# 2026-10-21 Hayai OCR provider ('hayai-ocr', on-device CJK manga OCR on onnxruntime): the desktop
# manga tab lists it, downloads / loads its model files, and the editor's region OCR crops for it
# like manga-ocr. The editor function lives in manga_editor_core since U8 (frozen in ImageRenderer).
_HAYAI_MANGA_INTEGRATION = [
    ("                        raise  # Re-raise to be caught by outer exception handler\n                        \n                elif provider == 'Qwen2-VL':\n",
     '                        raise  # Re-raise to be caught by outer exception handler\n\n                elif provider == \'hayai-ocr\':\n                    # Hayai OCR: the manga_models registry files (pinned revision, resumable,\n                    # sha256-verified) on a worker thread; Cancel / closing the dialog stops it\n                    # and keeps the partial file for the next attempt.\n                    import manga_models\n                    hayai_specs = manga_models.specs_for_ocr_provider(\'hayai-ocr\')\n                    grand_total = max(1, sum(spec.size for spec in hayai_specs))\n                    mb = 1024 * 1024\n                    hayai_cancel = threading.Event()\n                    cancel_btn.clicked.connect(hayai_cancel.set)\n                    download_dialog.finished.connect(lambda *_: hayai_cancel.set())\n                    cancel_btn.setText("Cancel")\n                    progress_label.setText("Downloading Hayai OCR model...")\n                    add_log(f"Downloading Hayai OCR v2.5 Nova ({manga_models.format_size(grand_total)}) "\n                            f"from Hugging Face ({hayai_specs[0].repo_id})")\n\n                    def download_hayai():\n                        done = 0\n                        try:\n                            for index, spec in enumerate(hayai_specs, 1):\n                                post_dialog_update(lambda i=index, s=spec: add_log(\n                                    f"[{i}/{len(hayai_specs)}] {s.filename} ({manga_models.format_size(s.size)})"))\n\n                                def report(p, base=done, title=spec.title):\n                                    def apply():\n                                        pct = int(min(99, (base + p.downloaded) * 100 / grand_total))\n                                        progress_bar.setValue(pct)\n                                        progress_label.setText(f"Downloading: {pct}%")\n                                        status_label.setText(f"{title}: {p.downloaded / mb:.1f}/{p.total / mb:.1f} MB")\n                                        size_label.setText(f"{(base + p.downloaded) / mb:.1f} MB / {grand_total / mb:.1f} MB")\n                                        if p.bytes_per_sec:\n                                            speed_label.setText(f"Speed: {p.bytes_per_sec / mb:.1f} MB/s")\n                                    post_dialog_update(apply)\n\n                                manga_models.download(spec, progress=report, cancel=hayai_cancel.is_set)\n                                done += spec.size\n\n                            def finished():\n                                progress_bar.setValue(100)\n                                progress_label.setText("✅ Hayai OCR downloaded")\n                                status_label.setText("Click \'Load Model\' (or start a run) to use it")\n                                speed_label.setText("")\n                                add_log("✅ Model files downloaded and verified")\n                                cancel_btn.setText("Close")\n                            post_dialog_update(finished)\n                        except Exception as hayai_error:\n                            def failed(err=hayai_error):\n                                progress_label.setText("⏹️ Download stopped" if hayai_cancel.is_set() else "❌ Download failed")\n                                add_log(f"{type(err).__name__}: {err}")\n                                cancel_btn.setText("Close")\n                            post_dialog_update(failed)\n                        try:\n                            self.update_queue.put((\'call_method\', self._check_provider_status, ()))\n                        except Exception:\n                            pass\n\n                    threading.Thread(target=download_hayai, name="HayaiOCRDownload", daemon=True).start()\n\n                elif provider == \'Qwen2-VL\':\n'),
    ('                self.provider_status_label.setStyleSheet("color: orange;")\n                self.provider_setup_btn.setText("Load Model")\n',
     '                self.provider_status_label.setStyleSheet("color: orange;")\n                if provider == \'hayai-ocr\' and provider_instance is not None and not provider_instance.models_downloaded():\n                    self.provider_status_label.setText("📥 Model not downloaded (downloads on load)")\n                self.provider_setup_btn.setText("Load Model")\n'),
    ('                self.provider_setup_btn.setVisible(True)\n                if provider in [\'Qwen2-VL\', \'manga-ocr\']:\n                    self.ocr_download_model_btn.setText("Download Model")\n',
     '                self.provider_setup_btn.setVisible(True)\n                if provider in [\'Qwen2-VL\', \'manga-ocr\', \'hayai-ocr\']:\n                    self.ocr_download_model_btn.setText("Download Model")\n'),
    ("                # Categorize providers\n                huggingface_providers = ['manga-ocr', 'Qwen2-VL', 'rapidocr']  # Move rapidocr here\n                pip_providers = ['easyocr', 'paddleocr', 'doctr']  # Remove rapidocr from here\n",
     "                # Categorize providers\n                huggingface_providers = ['manga-ocr', 'hayai-ocr', 'Qwen2-VL', 'rapidocr']  # Move rapidocr here\n                pip_providers = ['easyocr', 'paddleocr', 'doctr']  # Remove rapidocr from here\n"),
    ('                            success = False\n                else:\n',
     '                            success = False\n                elif provider == \'hayai-ocr\':\n                    # downloads the missing model files first (manga_models), with progress here\n                    success = ocr_manager.load_provider(provider, progress_callback=lambda pct, done_mb, total_mb, _speed: (\n                        update_progress(f"Downloading Hayai OCR: {done_mb:.0f}/{total_mb:.0f} MB", pct)))\n                else:\n'),
    ('            \'manga-ocr\': "Manga OCR - optimized for Japanese manga",\n            \'rapidocr\': "RapidOCR - fast local OCR with region detection",\n',
     '            \'manga-ocr\': "Manga OCR - optimized for Japanese manga",\n            \'hayai-ocr\': "Hayai OCR - local Japanese / Chinese / Korean / English bubble OCR (ONNX, no PyTorch)",\n            \'rapidocr\': "RapidOCR - fast local OCR with region detection",\n'),
    ("            ('manga-ocr', '🇯🇵 Manga OCR (Japanese)'),\n            ('Qwen2-VL', '🇰🇷 Qwen2-VL (Korean)'),\n",
     "            ('manga-ocr', '🇯🇵 Manga OCR (Japanese)'),\n            ('hayai-ocr', '🌏 Hayai OCR (CJK, Local)'),\n            ('Qwen2-VL', '🇰🇷 Qwen2-VL (Korean)'),\n"),
]
_HAYAI_IMAGERENDERER = [
    ('                \n        elif provider == \'manga-ocr\':\n            # For manga-ocr, process each region individually (cropped) for better accuracy\n            print(f"[OCR_REGIONS] Running manga-ocr OCR on cropped regions")\n            try:\n',
     '                \n        elif provider in (\'manga-ocr\', \'hayai-ocr\'):\n            # For manga-ocr / hayai-ocr (crop recognizers), process each region individually (cropped) for better accuracy\n            print(f"[OCR_REGIONS] Running {provider} OCR on cropped regions")\n            try:\n'),
    ('                if provider_obj is None:\n                    self._log(f"❌ OCR provider \'manga-ocr\' not available in OCRManager", "error")\n                    print(f"[OCR_REGIONS] Provider \'manga-ocr\' not found in OCRManager")\n                    return []\n',
     '                if provider_obj is None:\n                    self._log(f"❌ OCR provider \'{provider}\' not available in OCRManager", "error")\n                    print(f"[OCR_REGIONS] Provider \'{provider}\' not found in OCRManager")\n                    return []\n'),
    ('                if not provider_obj.is_loaded:\n                    print(f"[OCR_REGIONS] Loading OCR provider: manga-ocr")\n                    load_success = self.ocr_manager.load_provider(provider, **ocr_config)\n',
     '                if not provider_obj.is_loaded:\n                    print(f"[OCR_REGIONS] Loading OCR provider: {provider}")\n                    load_success = self.ocr_manager.load_provider(provider, **ocr_config)\n'),
    ('                    if not load_success:\n                        self._log(f"❌ Failed to load manga-ocr", "error")\n                        return []\n',
     '                    if not load_success:\n                        self._log(f"❌ Failed to load {provider}", "error")\n                        return []\n'),
    ('                    if _is_translation_cancelled(self):\n                        print(f"[OCR_REGIONS] Cancelled at manga-ocr region {i+1}/{len(regions)}")\n                        return []\n',
     '                    if _is_translation_cancelled(self):\n                        print(f"[OCR_REGIONS] Cancelled at {provider} region {i+1}/{len(regions)}")\n                        return []\n'),
    ('                \n                print(f"[OCR_REGIONS] manga-ocr recognized text in {len(recognized_texts)}/{len(regions)} regions")\n                return recognized_texts\n',
     '                \n                print(f"[OCR_REGIONS] {provider} recognized text in {len(recognized_texts)}/{len(regions)} regions")\n                return recognized_texts\n'),
    ('            except Exception as e:\n                print(f"[OCR_REGIONS] manga-ocr OCR error: {str(e)}")\n                import traceback\n',
     '            except Exception as e:\n                print(f"[OCR_REGIONS] {provider} OCR error: {str(e)}")\n                import traceback\n'),
    ('                import traceback\n                print(f"[OCR_REGIONS] manga-ocr OCR traceback: {traceback.format_exc()}")\n                self._log(f"❌ manga-ocr OCR failed: {str(e)}", "error")\n                return []\n',
     '                import traceback\n                print(f"[OCR_REGIONS] {provider} OCR traceback: {traceback.format_exc()}")\n                self._log(f"❌ {provider} OCR failed: {str(e)}", "error")\n                return []\n'),
]


# 2026-10-21 API call delay ranges: the main window's API call delay field takes a delay ("5") or a range
# ("5-7"; settings_rules.parse_delay_range); delay keeps the min and the new delay_max the max ('' =
# fixed); a config without delay (a new user) gets 3-7 (the default delay was 5). Exported as
# SEND_INTERVAL_SECONDS (the min) and SEND_INTERVAL_MAX_SECONDS (the max, '' = fixed).
_DELAY_ROW = (
    "            ('delay', ['delay_entry'], 5.0, lambda v: safe_float(v, 5.0)),\n",
    "            ('delay', ['delay_entry'], 3.0, delay_range_min),\n"
    "            ('delay_max', ['delay_entry'], 7.0, delay_range_max),\n")
_DELAY_ROW_GUI = (
    "                ('delay', ['delay_entry'], 5.0, lambda v: safe_float(v, 5.0)),\n",
    "                ('delay', ['delay_entry'], 3.0, delay_range_min),\n"
    "                ('delay_max', ['delay_entry'], 7.0, delay_range_max),\n")
_DELAY_ENV_DICT = (
    "            'SEND_INTERVAL_SECONDS': str(self.delay_entry.text()),\n",
    "            'SEND_INTERVAL_SECONDS': delay_env(self.delay_entry.text())[0],\n"
    "            'SEND_INTERVAL_MAX_SECONDS': delay_env(self.delay_entry.text())[1],\n")
# the field shows '5' / '5-7' (a config without delay: '3-7')
_DELAY_FIELD = ("        self.delay_entry.setText(str(self.config.get('delay', 5)))\n",
                "        self.delay_entry.setText(delay_range_text(self.config))\n")
# the names the rows above use (settings_rules)
# (on the `import os` line itself: the legacy line numbers the U7 span pins count stay as they were)
_DELAY_IMPORT = ("\nimport os\n", "\nimport os; from settings_rules import delay_env, delay_log_value, delay_range_max, "
                                  "delay_range_min, delay_range_text, owner_delay_env\n")
# the moved runners / dialogs that export or read the delay (one line each, so the U7 spans keep
# their line counts); the names come from _DELAY_IMPORT
_DELAY_RPGMAKER = [
    ("                os.environ['SEND_INTERVAL_SECONDS'] = str(self.delay_entry.text() or '2.0')\n",
     "                os.environ['SEND_INTERVAL_SECONDS'], os.environ['SEND_INTERVAL_MAX_SECONDS'] = owner_delay_env(self, '2.0')\n"),
    ("\n            os.environ['SEND_INTERVAL_SECONDS'] = str(self.delay_entry.text() or '2.0')\n",
     "\n            os.environ['SEND_INTERVAL_SECONDS'], os.environ['SEND_INTERVAL_MAX_SECONDS'] = owner_delay_env(self, '2.0')\n"),
]
_DELAY_GLOSSARY_LOG = (
    "            api_delay = float(self.delay_entry.text()) if hasattr(self, 'delay_entry') else 2.0\n",
    "            api_delay = delay_log_value(self.delay_entry.text()) if hasattr(self, 'delay_entry') else 2.0\n")
_DELAY_MANGA = (
    "                os.environ['SEND_INTERVAL_SECONDS'] = str(delay)\n",
    "                os.environ['SEND_INTERVAL_SECONDS'], os.environ['SEND_INTERVAL_MAX_SECONDS'] = delay_env(delay)\n")
_DELAY_ASYNC = (
    "        if hasattr(self.gui.delay_entry, 'get'):\n"
    "            env_vars['SEND_INTERVAL_SECONDS'] = str(self.gui.delay_entry.get())\n"
    "        else:\n"
    "            env_vars['SEND_INTERVAL_SECONDS'] = str(self.gui.delay_entry.text() if hasattr(self.gui.delay_entry, 'text') else '2')\n",
    "        # a range like 3-7 exports its min and max (settings_rules.owner_delay_env)\n"
    "        from settings_rules import owner_delay_env\n"
    "        env_vars['SEND_INTERVAL_SECONDS'], env_vars['SEND_INTERVAL_MAX_SECONDS'] = owner_delay_env(self.gui, '2')\n")
# the QA dialog's AI-truncation delay field takes '5' / '5-7' / -1 (QLineEdit; was a spin box) and the
# scanner passes the range (inherit: -1 = the global delay and its range)
_DELAY_QA_GUI = [('from language_options import TARGET_LANGUAGES\nfrom emoticon_patterns import DEFAULT_EMOTICON_PATTERNS\n',
  'from language_options import TARGET_LANGUAGES\n'
  'from settings_rules import delay_field_text, delay_field_values, format_delay_field\n'
  'from emoticon_patterns import DEFAULT_EMOTICON_PATTERNS\n'),
 ('\n'
  '            ai_delay_spin = QDoubleSpinBox()\n'
  '            ai_delay_spin.setMinimum(-1.0)\n'
  '            ai_delay_spin.setMaximum(60.0)\n'
  '            ai_delay_spin.setSingleStep(0.5)\n'
  "            ai_delay_spin.setValue(float(qa_settings.get('ai_truncation_api_call_delay', -1.0)))\n"
  '            ai_delay_spin.setFixedWidth(100)\n'
  '            ai_delay_spin.setToolTip("-1 = use global API call delay")\n'
  '            disable_wheel_event(ai_delay_spin)\n'
  '            ai_delay_h.addWidget(ai_delay_spin)\n'
  '\n'
  '            # Snap negative values in-between 0 and -1 directly\n'
  '            ai_delay_spin.setProperty("last_val", ai_delay_spin.value())\n'
  '            def enforce_delay_snap(val):\n'
  '                last_val = ai_delay_spin.property("last_val")\n'
  '                ai_delay_spin.blockSignals(True)\n'
  '                if -1.0 < val < 0.0:\n'
  '                    if last_val >= 0.0:\n'
  '                        val = -1.0\n'
  '                    elif last_val <= -1.0:\n'
  '                        val = 0.0\n'
  '                    else:\n'
  '                        val = -1.0\n'
  '                    ai_delay_spin.setValue(val)\n'
  '                ai_delay_spin.setProperty("last_val", val)\n'
  '                ai_delay_spin.blockSignals(False)\n'
  '            ai_delay_spin.valueChanged.connect(enforce_delay_snap)\n'
  '\n'
  '            ai_delay_hint = QLabel("sec  (-1 = global)")\n'
  "            ai_delay_hint.setFont(QFont('Arial', 8))\n",
  '\n'
  '            # seconds, or a range like 5-7 (each call waits a random time between the two); -1 = global\n'
  '            from PySide6.QtWidgets import QLineEdit\n'
  '            ai_delay_spin = QLineEdit()\n'
  "            _qa_delay_max = qa_settings.get('ai_truncation_api_call_delay_max', '')\n"
  '            ai_delay_spin.setText(format_delay_field(\n'
  "                qa_settings.get('ai_truncation_api_call_delay', -1.0), _qa_delay_max, inherit=-1.0))\n"
  '            ai_delay_spin.setFixedWidth(100)\n'
  '            ai_delay_spin.setToolTip("-1 = use global API call delay (and its range)\\n"\n'
  '                                     "A range like 5-7 waits a random time between the two before each call")\n'
  '            ai_delay_h.addWidget(ai_delay_spin)\n'
  '\n'
  '            def _qa_delay_values(widget):\n'
  "                values = delay_field_values(widget.text(), 'ai_truncation_api_call_delay', inherit=-1.0)\n"
  '                if values is None:\n'
  '                    raise ValueError(f"not a delay: {widget.text()!r}")\n'
  '                return values\n'
  '\n'
  '            def normalize_delay_text():\n'
  "                values = delay_field_values(ai_delay_spin.text(), 'ai_truncation_api_call_delay', inherit=-1.0)\n"
  '                if values is not None:\n'
  "                    ai_delay_spin.setText(delay_field_text(values, 'ai_truncation_api_call_delay', "
  'inherit=-1.0))\n'
  '            ai_delay_spin.editingFinished.connect(normalize_delay_text)\n'
  '\n'
  '            ai_delay_hint = QLabel("sec or 5-7  (-1 = global)")\n'
  "            ai_delay_hint.setFont(QFont('Arial', 8))\n"),
 ("                        'ai_truncation_max_tokens': (ai_tokens_spin, lambda x: x.value()),\n"
  "                        'ai_truncation_api_call_delay': (ai_delay_spin, lambda x: x.value()),\n"
  "                        'ai_truncation_endpoint_url': (ai_url_entry, lambda x: x.text().strip()),\n",
  "                        'ai_truncation_max_tokens': (ai_tokens_spin, lambda x: x.value()),\n"
  "                        'ai_truncation_api_call_delay': (ai_delay_spin, lambda x: "
  "_qa_delay_values(x)['ai_truncation_api_call_delay']),\n"
  "                        'ai_truncation_api_call_delay_max': (ai_delay_spin, lambda x: "
  "_qa_delay_values(x)['ai_truncation_api_call_delay_max']),\n"
  "                        'ai_truncation_endpoint_url': (ai_url_entry, lambda x: x.text().strip()),\n"),
 ("                    ('ai_truncation_max_tokens', ai_tokens_spin, 0),\n"
  "                    ('ai_truncation_api_call_delay', ai_delay_spin, 0),\n"
  "                    ('ai_truncation_endpoint_url', ai_url_entry, ''),\n",
  "                    ('ai_truncation_max_tokens', ai_tokens_spin, 0),\n"
  "                    ('ai_truncation_endpoint_url', ai_url_entry, ''),\n"),
 _after("                for key, widget, default in simple:\n"
        "                    _apply(lambda k=key, w=widget, d=default: _set_value(w, s.get(k, d)))",
        "                # the AI truncation API call delay field: '5', a range '5-7', or -1 (global)",
        "                _apply(lambda: ai_delay_spin.setText(delay_field_text(",
        "                    {'ai_truncation_api_call_delay': s.get('ai_truncation_api_call_delay', 0),",
        "                     'ai_truncation_api_call_delay_max': s.get('ai_truncation_api_call_delay_max', '')},",
        "                    'ai_truncation_api_call_delay', inherit=-1.0)))")]
_DELAY_SCANNER = [('@contextmanager\n'
  'def _qa_api_overrides_env(disable_thinking=False, delay_override=-1.0):\n'
  '    """Temporarily overrides API config environment variables for QA scans."""\n',
  '@contextmanager\n'
  'def _qa_api_overrides_env(disable_thinking=False, delay_override=-1.0, delay_max_override=None):\n'
  '    """Temporarily overrides API config environment variables for QA scans."""\n'),
 ("    orig_delay = os.environ.get('SEND_INTERVAL_SECONDS')\n    \n",
  "    orig_delay = os.environ.get('SEND_INTERVAL_SECONDS')\n"
  "    orig_delay_max = os.environ.get('SEND_INTERVAL_MAX_SECONDS')\n"
  '    \n'),
 ("        os.environ['SEND_INTERVAL_SECONDS'] = str(delay_override)\n        \n",
  "        os.environ['SEND_INTERVAL_SECONDS'] = str(delay_override)\n"
  "        # the QA delay's range max ('' = a fixed delay)\n"
  "        os.environ['SEND_INTERVAL_MAX_SECONDS'] = str(delay_max_override) if delay_max_override else ''\n"
  '        \n'),
 ("        elif delay_override >= 0: os.environ.pop('SEND_INTERVAL_SECONDS', None)\n\n",
  "        elif delay_override >= 0: os.environ.pop('SEND_INTERVAL_SECONDS', None)\n"
  "        if orig_delay_max is not None: os.environ['SEND_INTERVAL_MAX_SECONDS'] = orig_delay_max\n"
  "        elif delay_override >= 0: os.environ.pop('SEND_INTERVAL_MAX_SECONDS', None)\n"
  '\n'),
 ('                            custom_system_prompt=None, temperature=None, max_tokens=None,\n'
  "                            prompt_role='system', disable_thinking=False, api_call_delay=-1.0):\n"
  '    """Check if translated content appears truncated by asking an AI model.\n',
  '                            custom_system_prompt=None, temperature=None, max_tokens=None,\n'
  "                            prompt_role='system', disable_thinking=False, api_call_delay=-1.0,\n"
  '                            api_call_delay_max=None):\n'
  '    """Check if translated content appears truncated by asking an AI model.\n'),
 ('        \n'
  '        with _qa_api_overrides_env(disable_thinking=disable_thinking, delay_override=api_call_delay):\n'
  "            # Snapshot the QA-specific delay into the client's thread-local\n",
  '        \n'
  '        _delay_max = api_call_delay_max if api_call_delay >= 0 and api_call_delay_max and '
  'float(api_call_delay_max) > api_call_delay else None\n'
  '        with _qa_api_overrides_env(disable_thinking=disable_thinking, delay_override=api_call_delay,\n'
  '                                   delay_max_override=_delay_max):\n'
  "            # Snapshot the QA-specific delay into the client's thread-local\n"),
 ('                    _tls.active_api_delay_override = float(api_call_delay)\n                except Exception:\n',
  '                    _tls.active_api_delay_override = float(api_call_delay)\n'
  '                    _tls.active_api_delay_max = float(_delay_max) if _delay_max else None\n'
  '                except Exception:\n'),
 ('                    client._qa_truncation_delay_override = float(api_call_delay)\n'
  '                except Exception:\n',
  '                    client._qa_truncation_delay_override = float(api_call_delay)\n'
  '                    client._qa_truncation_delay_max_override = float(_delay_max) if _delay_max else None\n'
  '                except Exception:\n'),
 ('                    # Read delay: dedicated setting (-1 = global, else use value)\n'
  "                    _dedicated_delay = float(qa_settings.get('ai_truncation_api_call_delay', -1.0))\n",
  '                    # Read delay: dedicated setting (-1 = global, else use value)\n'
  '                    # a range (min, max): the dedicated one, or -1 = the global delay and its range\n'
  '                    from settings_rules import delay_range_for\n'
  "                    _dedicated_delay = float(qa_settings.get('ai_truncation_api_call_delay', -1.0))\n"),
 ('                        # -1 means use global API call delay from translator config\n'
  '                        try:\n'
  "                            _ai_delay = float(_ai_config.get('delay', 2))\n"
  '                        except (ValueError, TypeError):\n'
  '                            _ai_delay = 2.0\n'
  '                    else:\n'
  '                        _ai_delay = _dedicated_delay\n'
  '\n',
  '                        # -1 means use global API call delay from translator config\n'
  '                        _ai_delay, _ai_delay_max = delay_range_for(_ai_config)\n'
  '                    else:\n'
  '                        _ai_delay, _ai_delay_max = delay_range_for(\n'
  "                            qa_settings, 'ai_truncation_api_call_delay', default_max=None)\n"
  '\n'),
 ('                        api_call_delay=_ai_delay,\n                    )\n\n',
  '                        api_call_delay=_ai_delay,\n'
  '                        api_call_delay_max=_ai_delay_max,\n'
  '                    )\n'
  '\n')]

# 2026-10-22 Clean and Translate load the selected local inpainter when no preload filled its pool
# (the model was switched after the startup preload): Clean polled for 30 s a pool nothing would fill,
# then failed. _load_inpainter_into_pool (Translate's last-resort load, lifted out) is shared by both.
_POOL_LOAD_HELPER = (
    '    return regions\n'
    'def _run_clean_background(self, image_path: str, regions: list):\n',
    '    return regions\n'
    'def _load_inpainter_into_pool(self, local_model: str, resolved_model_path: str, key, is_custom_image_edit: bool):\n'
    '    """Load a new inpainter and add it to the shared pool under *key* as a spare (not checked out).\n'
    '    Used when no preload is filling the pool, e.g. the model was switched after startup preload."""\n'
    '    from manga_translator import MangaTranslator\n'
    '    from local_inpainter import LocalInpainter\n'
    '    new_inpainter = LocalInpainter()\n'
    '    try:\n'
    "        if is_custom_image_edit and hasattr(self, 'main_gui') and getattr(self.main_gui, 'config', None):\n"
    '            new_inpainter.config.update(self.main_gui.config)\n'
    "            default_endpoint = ''\n"
    '            use_custom_openai = bool(\n'
    "                getattr(self.main_gui, 'use_custom_openai_endpoint_var', False)\n"
    "                or self.main_gui.config.get('use_custom_openai_endpoint', False)\n"
    '            )\n'
    '            if use_custom_openai:\n'
    "                default_endpoint = self.main_gui.config.get('openai_base_url', '')\n"
    "                default_endpoint = default_endpoint or getattr(self.main_gui, 'openai_base_url_var', '') or ''\n"
    "            new_inpainter.config['custom_image_edit_default_endpoint'] = default_endpoint\n"
    "            new_inpainter.config['openai_base_url'] = default_endpoint\n"
    "            new_inpainter.config['use_custom_openai_endpoint'] = use_custom_openai\n"
    '    except Exception:\n'
    '        pass\n'
    '    if not new_inpainter.load_model(local_model, resolved_model_path):\n'
    '        return None\n'
    '    with MangaTranslator._inpaint_pool_lock:\n'
    '        rec = MangaTranslator._inpaint_pool.get(key)\n'
    '        if not rec:\n'
    '            rec = MangaTranslator._inpaint_pool[key] = {\n'
    "                'spares': [],\n"
    "                'checked_out': [],\n"
    "                'model_type': local_model,\n"
    "                'model_path': resolved_model_path\n"
    '            }\n'
    "        rec.setdefault('spares', []).append(new_inpainter)\n"
    "        rec.setdefault('checked_out', [])\n"
    '    return new_inpainter\n'
    'def _run_clean_background(self, image_path: str, regions: list):\n')
_POOL_LOAD_CLEAN = (
    '                    # POLL for inpainter with timeout (same as translator initialization)\n'
    '                    inpainter = None\n'
    '                    poll_timeout = 30  # 30 seconds\n',
    '                    # No pool for this model (no preload ran for it, e.g. the model was switched after\n'
    '                    # startup): load one into the pool now instead of polling a pool nothing will fill.\n'
    '                    poll_timeout = 30  # 30 seconds\n'
    '                    if not is_custom_image_edit:\n'
    '                        pool_key = (local_model, os.path.abspath(os.path.normpath(resolved_model_path)))\n'
    '                        with MangaTranslator._inpaint_pool_lock:\n'
    '                            has_pool = pool_key in MangaTranslator._inpaint_pool\n'
    '                        if not has_pool:\n'
    '                            self._log(f"📥 Loading {local_model} inpainter...", "info")\n'
    '                            if not _load_inpainter_into_pool(self, local_model, pool_key[1], pool_key, False):\n'
    '                                self._log(f"❌ Failed to load {local_model} inpainter", "error")\n'
    '                                poll_timeout = 0\n'
    '\n'
    '                    # POLL for inpainter with timeout (same as translator initialization)\n'
    '                    inpainter = None\n')
# 2026-10-22 a batch Start leaves the manual editor's boxes: the automatic OCR save also writes the
# page's editor state (manga_env._record_page_editor_state)
_BATCH_EDITOR_STATE_NO_DOC = (
    "        if not document or not output_path:\n            return\n",
    "        if not document or not output_path:\n"
    "            _record_page_editor_state(self, image_path, manga_ocr_io.make_page(image_path, regions, index=1))\n"
    "            return\n",
)
_BATCH_EDITOR_STATE_DOC = (
    "            manga_ocr_io.write_document(output_path, document)\n        translated_count = sum(\n",
    "            manga_ocr_io.write_document(output_path, document)\n"
    "        _record_page_editor_state(self, image_path, page)\n        translated_count = sum(\n",
)

_POOL_LOAD_SYNC = (
    '                        new_inpainter = LocalInpainter()\n'
    '                        try:\n'
    "                            if is_custom_image_edit and hasattr(self, 'main_gui') and getattr(self.main_gui, 'config', None):\n"
    '                                new_inpainter.config.update(self.main_gui.config)\n'
    "                                default_endpoint = ''\n"
    '                                use_custom_openai = bool(\n'
    "                                    getattr(self.main_gui, 'use_custom_openai_endpoint_var', False)\n"
    "                                    or self.main_gui.config.get('use_custom_openai_endpoint', False)\n"
    '                                )\n'
    '                                if use_custom_openai:\n'
    "                                    default_endpoint = self.main_gui.config.get('openai_base_url', '')\n"
    "                                    default_endpoint = default_endpoint or getattr(self.main_gui, 'openai_base_url_var', '') or ''\n"
    "                                new_inpainter.config['custom_image_edit_default_endpoint'] = default_endpoint\n"
    "                                new_inpainter.config['openai_base_url'] = default_endpoint\n"
    "                                new_inpainter.config['use_custom_openai_endpoint'] = use_custom_openai\n"
    '                        except Exception:\n'
    '                            pass\n'
    '                        if new_inpainter.load_model(local_model, resolved_model_path):\n'
    '                            with MangaTranslator._inpaint_pool_lock:\n'
    '                                rec = MangaTranslator._inpaint_pool.get(key)\n'
    '                                if not rec:\n'
    '                                    MangaTranslator._inpaint_pool[key] = {\n'
    "                                        'spares': [],\n"
    "                                        'checked_out': [],\n"
    "                                        'model_type': local_model,\n"
    "                                        'model_path': resolved_model_path\n"
    '                                    }\n'
    '                                    rec = MangaTranslator._inpaint_pool[key]\n'
    "                                rec['spares'].append(new_inpainter)\n"
    "                                rec['checked_out'].append(new_inpainter)\n"
    '                                inpainter = new_inpainter\n',
    '                        new_inpainter = _load_inpainter_into_pool(self, local_model, resolved_model_path, key, is_custom_image_edit)\n'
    '                        if new_inpainter:\n'
    '                            with MangaTranslator._inpaint_pool_lock:\n'
    "                                MangaTranslator._inpaint_pool[key]['checked_out'].append(new_inpainter)\n"
    '                            inpainter = new_inpainter\n')



# 2026-10-22 The image state manager's worker shutdown no longer waits for its queues' feeder
# threads (cancel_join_thread): with the worker gone a feeder writing a queued state blocked
# forever and the window hung on close. Every caller flush()es the state itself first.
_STATE_QUEUE_CLOSE = (
    '            # Close queues to release handles\n'
    '            if self._mp_task_q:\n'
    '                try:\n'
    '                    self._mp_task_q.close()\n'
    '                    self._mp_task_q.join_thread()\n'
    '                except Exception:\n'
    '                    pass\n'
    '            \n'
    '            if self._mp_result_q:\n'
    '                try:\n'
    '                    self._mp_result_q.close()\n'
    '                    self._mp_result_q.join_thread()\n',
    '            # Close queues to release handles. Never wait for their feeder threads: with the\n'
    '            # worker gone nobody reads the pipe, so a feeder still writing a queued state blocks\n'
    '            # forever (the window hung on close). Every caller flush()es the state itself first.\n'
    '            if self._mp_task_q:\n'
    '                try:\n'
    '                    self._mp_task_q.cancel_join_thread()\n'
    '                    self._mp_task_q.close()\n'
    '                except Exception:\n'
    '                    pass\n'
    '\n'
    '            if self._mp_result_q:\n'
    '                try:\n'
    '                    self._mp_result_q.cancel_join_thread()\n'
    '                    self._mp_result_q.close()\n')


# 2026-10-22 "Glossary Name Check After Translation (Experimental)" (glossary_name_check /
# GLOSSARY_NAME_CHECK, off; glossary_name_check_mode / GLOSSARY_NAME_CHECK_MODE, 'auto') and the QA
# scanner's "Check glossary names (Experimental)" (check_glossary_names / QA_CHECK_GLOSSARY_NAMES, off)
_NAME_CHECK_SETTINGS_ROWS = (
    "('glossary_name_check', ['glossary_name_check_var'], False, bool),",
    "('glossary_name_check_mode', ['glossary_name_check_mode_var'], 'auto', str),",
    "('glossary_name_check_min_chars', ['glossary_name_check_min_chars_var'], 2, lambda v: max(0, min(10, "
    "safe_int(v, 2)))),",
    "('glossary_name_check_autofix_level', ['glossary_name_check_autofix_level_var'], 1, lambda v: max(1, min(3, "
    "safe_int(v, 1)))),",
    "('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),",
)
_NAME_CHECK_MIN_CHARS_ROW = ("('emergency_glossary_compliance_min_chars', ['emergency_glossary_compliance_min_chars_var'], "
                             "3, lambda v: max(0, min(10, safe_int(v, 3)))),")
_NAME_CHECK_SETTINGS_ROW = _after(
    "            " + _NAME_CHECK_MIN_CHARS_ROW, *("            " + row for row in _NAME_CHECK_SETTINGS_ROWS))
_NAME_CHECK_SETTINGS_ROW_GUI = _after(
    "                " + _NAME_CHECK_MIN_CHARS_ROW, *("                " + row for row in _NAME_CHECK_SETTINGS_ROWS))
_NAME_CHECK_BOOL_ROW = _after(
    "            ('emergency_glossary_compliance_var', 'emergency_glossary_compliance', False),",
    "            ('glossary_name_check_var', 'glossary_name_check', False),",
    "            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),")
_NAME_CHECK_STR_ROW = _after(
    "            ('emergency_glossary_compliance_mode_var', 'emergency_glossary_compliance_mode', 'characters'),",
    "            ('glossary_name_check_mode_var', 'glossary_name_check_mode', 'auto'),")
_NAME_CHECK_ENV_DICT = _after(
    "            'EMERGENCY_GLOSSARY_COMPLIANCE_MIN_CHARS': str(getattr(self, "
    "'emergency_glossary_compliance_min_chars_var', 3)),",
    "            'GLOSSARY_NAME_CHECK': \"1\" if (getattr(self, 'glossary_name_check_var', False) and "
    "auto_glossary_mode != 'no_glossary') else \"0\",",
    "            'GLOSSARY_NAME_CHECK_MODE': str(getattr(self, 'glossary_name_check_mode_var', 'auto')),",
    "            'GLOSSARY_NAME_CHECK_MIN_CHARS': str(getattr(self, 'glossary_name_check_min_chars_var', 2)),",
    "            'GLOSSARY_NAME_CHECK_AUTOFIX_LEVEL': str(getattr(self, 'glossary_name_check_autofix_level_var', 1)),",
    "            'GLOSSARY_NAME_CHECK_USE_NER': \"1\" if getattr(self, 'glossary_name_check_use_ner_var', True) "
    "else \"0\",")
# the name check's own minimum raw entry length, read at startup like Emergency Glossary Compliance's
_NAME_CHECK_MIN_CHARS_INIT = _after(
    "            self.emergency_glossary_compliance_min_chars_var = 3",
    "        try:",
    "            self.glossary_name_check_min_chars_var = max(0, min(10, int(self.config.get("
    "'glossary_name_check_min_chars', 2))))",
    "        except (TypeError, ValueError):",
    "            self.glossary_name_check_min_chars_var = 2",
    "        try:",
    "            self.glossary_name_check_autofix_level_var = max(1, min(3, int(self.config.get("
    "'glossary_name_check_autofix_level', 1))))",
    "        except (TypeError, ValueError):",
    "            self.glossary_name_check_autofix_level_var = 1")
_NAME_CHECK_QA_LABEL = _after(
    "            'QA_CHECK_GLOSSARY_LEAKAGE': 'Check for glossary leakage',",
    "            'QA_CHECK_GLOSSARY_NAMES': 'Check glossary names (experimental)',")
_NAME_CHECK_QA_ENV_PAIR = _after(
    "                ('QA_CHECK_GLOSSARY_LEAKAGE', '1' if qa_settings.get('check_glossary_leakage', True) else '0'),",
    "                ('QA_CHECK_GLOSSARY_NAMES', '1' if qa_settings.get('check_glossary_names', False) else '0'),")
# the QA scanner pieces, as edited (dialog toggle + fix dialog, runtime default/env/scope, scanner step)
_NAME_CHECK_QA_GUI = [('\n    def prewarm_qa_scanner_gui(self):\n',
  '\n'
  '    def _glossary_name_message(self, parent, title, text, icon=QMessageBox.Information,\n'
  '                               buttons=QMessageBox.Ok):\n'
  '        """A name-check message box with centered, wider buttons; returns the button clicked."""\n'
  '        box = QMessageBox(icon, title, text, buttons, parent)\n'
  '        try:\n'
  '            from qt_widget_helpers import center_messagebox_buttons\n'
  '            center_messagebox_buttons(box, min_button_width=120, min_button_height=36)\n'
  '        except Exception:\n'
  '            pass\n'
  '        return box.exec()\n'
  '\n'
  '    def _glossary_name_row_menu(self, review, tree, pos, rows):\n'
  '        """Right-click on a review row: open the file, or preview where the name occurs."""\n'
  '        from PySide6.QtWidgets import QMenu\n'
  '        item = tree.itemAt(pos)\n'
  '        row = next((r for r in rows if r[0] is item), None)\n'
  '        if row is None:\n'
  '            return\n'
  '        _item, folder, output_file, finding = row\n'
  '        path = os.path.join(folder, output_file)\n'
  '        from qt_widget_helpers import CONTEXT_MENU_STYLE\n'
  '        menu = QMenu(tree)\n'
  '        menu.setStyleSheet(CONTEXT_MENU_STYLE)\n'
  "        is_html = output_file.lower().endswith(('.html', '.xhtml', '.htm'))\n"
  '        open_action = menu.addAction("📂 Open HTML file" if is_html else "📂 Open file")\n'
  '        preview_action = menu.addAction("🔍 Preview")\n'
  '        chosen = menu.exec(tree.viewport().mapToGlobal(pos))\n'
  '        if chosen is open_action:\n'
  '            from PySide6.QtGui import QDesktopServices\n'
  '            from PySide6.QtCore import QUrl\n'
  '            if not os.path.isfile(path) or not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):\n'
  '                self._glossary_name_message(review, "Open file", f"Could not open:\\n{path}", '
  'QMessageBox.Warning)\n'
  '        elif chosen is preview_action:\n'
  '            self._glossary_name_preview(review, folder, output_file, finding)\n'
  '\n'
  '    def _glossary_name_preview(self, parent, folder, output_file, finding):\n'
  '        """Where a reviewed name occurs: the translation (the variant highlighted) and, for a chapter,\n'
  '        the source chapter (the raw name highlighted)."""\n'
  '        import glossary_name_check as name_check\n'
  '        from PySide6.QtWidgets import QTextBrowser\n'
  '        try:\n'
  '            with open(os.path.join(folder, output_file), "r", encoding="utf-8", errors="ignore") as f:\n'
  '                markup = f.read()\n'
  '        except OSError as e:\n'
  '            self._glossary_name_message(parent, "Preview", str(e), QMessageBox.Warning)\n'
  '            return\n'
  '        if output_file in name_check.TITLE_FILES:\n'
  '            translated, source = name_check.finding_snippets(_html.escape(markup), finding), []\n'
  '        else:\n'
  '            translated = name_check.finding_snippets(markup, finding)\n'
  '            source = '
  'name_check.source_snippets(name_check.source_chapters_for_folder(folder).get(output_file, ""),\n'
  '                                                finding)\n'
  '\n'
  '        def lines(snippets, color):\n'
  '            if not snippets:\n'
  '                return "<p style=\'color:#888\'>None found.</p>"\n'
  '            return "".join(\n'
  '                f"<p style=\'margin:4px 0 8px 0\'>{_html.escape(before)}"\n'
  '                f"<span style=\'color:{color}; font-weight:bold\'>{_html.escape(found)}</span>"\n'
  '                f"{_html.escape(after)}</p>" for before, found, after in snippets)\n'
  '\n'
  '        origin = finding.origin or "Book glossary"\n'
  '        body = (f"<p><b>{_html.escape(\' / \'.join(finding.variants))}</b> → "\n'
  '                f"<b style=\'color:#5a9fd4\'>{_html.escape(finding.expected)}</b> "\n'
  '                f"<span style=\'color:#888\'>({_html.escape(finding.raw)}, '
  '{_html.escape(finding.entry_type)}, "\n'
  '                f"{_html.escape(origin)})</span></p>"\n'
  '                f"<h4>In the translation ({len(translated)})</h4>{lines(translated, \'#f0ad4e\')}")\n'
  '        if output_file not in name_check.TITLE_FILES:\n'
  '            body += f"<h4>In the source chapter ({len(source)})</h4>{lines(source, \'#5a9fd4\')}"\n'
  '        dialog = QDialog(parent)\n'
  '        dialog.setWindowTitle(f"Preview — {output_file}")\n'
  '        dialog.resize(760, 520)\n'
  '        layout = QVBoxLayout(dialog)\n'
  '        layout.setContentsMargins(18, 16, 18, 14)\n'
  '        layout.setSpacing(10)\n'
  '        view = QTextBrowser()\n'
  '        view.setOpenLinks(False)\n'
  '        view.setHtml(body)\n'
  '        layout.addWidget(view, 1)\n'
  '        close_row = QHBoxLayout()\n'
  '        close_btn = QPushButton("Close")\n'
  '        close_btn.setMinimumWidth(120)\n'
  '        close_btn.clicked.connect(dialog.accept)\n'
  '        close_row.addStretch()\n'
  '        close_row.addWidget(close_btn)\n'
  '        close_row.addStretch()\n'
  '        layout.addLayout(close_row)\n'
  '        dialog.exec()\n'
  '\n'
  '    def _glossary_name_fix_folders(self, parent):\n'
  '        """The output folders to check: with the QA scan\'s Auto-search output on, the output folder of\n'
  '        every selected input file; else (or when none is found) the folder picker."""\n'
  '        caption = "Select Output Folder to Check Glossary Names"\n'
  '        if self._qa_auto_search_enabled():\n'
  '            sources, _stored_source, _direct_text_only = self._qa_auto_search_sources()\n'
  '            folders = self._auto_search_output_folders(sources) if sources else []\n'
  '            if folders:\n'
  '                return list(dict.fromkeys(folders))\n'
  '            caption = "Auto-search failed - " + caption\n'
  "        start = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory') or os.getcwd()\n"
  '        folder = _normalize_qa_dialog_path(QFileDialog.getExistingDirectory(parent, caption, start))\n'
  '        return [folder] if folder else []\n'
  '\n'
  '    def _glossary_name_review_with_progress(self, parent, title, folders, use_ner):\n'
  '        """``review_folder`` for each folder on a worker thread behind a progress dialog (the NER model\n'
  '        download, then the chapters, several at once with NER: QA scanner › Maximum parallel workers),\n'
  '        so the window keeps responding. Returns the ``FolderReview`` list; None when cancelled."""\n'
  '        import glossary_name_check as name_check\n'
  '        scope = self._glossary_name_check_scope()\n'
  "        # only the glossaries a translation request uses: the book's, then Add Additional Glossary's\n"
  "        # extension and Enable Unified Glossary's shared file when they are on\n"
  '        layer_settings = name_check.layer_settings_from_config(self.config)\n'
  '        try:\n'
  '            layer_settings.update(self._unified_glossary_env_dict())  # the live checkboxes\n'
  '        except Exception:\n'
  '            pass\n'
  "        log = getattr(self, 'append_log', None)\n"
  '        mb = 1024 * 1024\n'
  '        cancel = threading.Event()\n'
  '        logs = []\n'
  '        state = {"text": "Checking glossary names...", "value": 0, "maximum": 0, "done": False,\n'
  '                 "result": None, "error": None}\n'
  '\n'
  '        def on_download(report):\n'
  '            speed = f" · {report.bytes_per_sec / mb:.1f} MB/s" if report.bytes_per_sec else ""\n'
  '            text = ("Verifying the named-entity model (NER)..." if report.phase == "verify" else\n'
  '                    f"Downloading the named-entity model (NER): {report.downloaded / mb:.1f} / "\n'
  '                    f"{report.total / mb:.1f} MB{speed}")\n'
  '            state.update(text=text, value=report.percent, maximum=100)\n'
  '\n'
  '        where = {"text": ""}\n'
  '\n'
  '        def on_chapter(done, total):\n'
  '            state.update(text=f"{where[\'text\']}Checking chapter {done + 1} of {total}...", value=done, '
  'maximum=total)\n'
  '\n'
  '        def work():\n'
  '            try:\n'
  '                ner = name_check.resolve_ner(use_ner, logs.append, progress=on_download, '
  'cancel=cancel.is_set,\n'
  '                                             retry=True)\n'
  '                reviews = []\n'
  '                for index, folder in enumerate(folders):\n'
  '                    if cancel.is_set():\n'
  '                        break\n'
  '                    if len(folders) > 1:\n'
  '                        where["text"] = f"{os.path.basename(folder)} ({index + 1} of {len(folders)})\\n"\n'
  '                    state.update(text=f"{where[\'text\']}Checking glossary names...", value=0, '
  'maximum=0)\n'
  '                    reviews.append(name_check.review_folder(folder, scope, stop=cancel.is_set, ner=ner,\n'
  '                                                            progress=on_chapter, '
  'settings=layer_settings))\n'
  '                state["result"] = reviews\n'
  '            except Exception as e:\n'
  '                state["error"] = e\n'
  '            finally:\n'
  '                state["done"] = True\n'
  '\n'
  '        # label, bar and a centered, wider Cancel (QProgressDialog keeps its button at the right)\n'
  '        from PySide6.QtWidgets import QProgressBar\n'
  '        dialog = QDialog(parent)\n'
  '        dialog.setWindowTitle(title)\n'
  '        dialog.setWindowModality(Qt.WindowModal)\n'
  '        dialog.setMinimumWidth(460)\n'
  '        dialog_layout = QVBoxLayout(dialog)\n'
  '        dialog_layout.setContentsMargins(18, 16, 18, 14)\n'
  '        dialog_layout.setSpacing(10)\n'
  '        label = QLabel(state["text"])\n'
  '        label.setAlignment(Qt.AlignCenter)\n'
  '        label.setWordWrap(True)\n'
  '        dialog_layout.addWidget(label)\n'
  '        bar = QProgressBar()\n'
  '        bar.setRange(0, 0)\n'
  '        dialog_layout.addWidget(bar)\n'
  '        cancel_row = QHBoxLayout()\n'
  '        cancel_button = QPushButton("Cancel")\n'
  '        cancel_button.setMinimumWidth(120)\n'
  '        cancel_button.setMinimumHeight(36)\n'
  '        cancel_row.addStretch()\n'
  '        cancel_row.addWidget(cancel_button)\n'
  '        cancel_row.addStretch()\n'
  '        dialog_layout.addLayout(cancel_row)\n'
  '\n'
  '        def request_cancel(*_args):\n'
  '            cancel.set()\n'
  '            cancel_button.setEnabled(False)\n'
  '            label.setText("Stopping...")\n'
  '\n'
  '        cancel_button.clicked.connect(request_cancel)\n'
  "        dialog.rejected.connect(request_cancel)  # Esc / the window's close button\n"
  '        loop = QEventLoop()\n'
  '        timer = QTimer()\n'
  '        timer.setInterval(100)\n'
  '\n'
  '        def tick():\n'
  '            while logs:\n'
  '                line = logs.pop(0)\n'
  '                if log:\n'
  '                    log(line)\n'
  '            if not cancel.is_set():\n'
  '                label.setText(state["text"])\n'
  '                bar.setRange(0, state["maximum"])\n'
  '                bar.setValue(state["value"])\n'
  '            if state["done"]:\n'
  '                timer.stop()\n'
  '                loop.quit()\n'
  '\n'
  '        timer.timeout.connect(tick)\n'
  '        threading.Thread(target=work, name="GlossaryNameReview", daemon=True).start()\n'
  '        timer.start()\n'
  '        dialog.show()\n'
  '        loop.exec()\n'
  '        dialog.rejected.disconnect(request_cancel)\n'
  '        dialog.close()\n'
  '        if cancel.is_set():\n'
  '            return None\n'
  '        if state["error"] is not None:\n'
  '            raise state["error"]\n'
  '        return state["result"]\n'
  '\n'
  '    def _glossary_name_check_scope(self):\n'
  '        """Emergency Glossary Compliance\'s entry types and the name check\'s minimum length (live '
  'values)."""\n'
  '        import glossary_name_check as name_check\n'
  '        config = dict(self.config or {})\n'
  "        for key in ('emergency_glossary_compliance_mode', 'emergency_glossary_compliance_custom_types',\n"
  "                    'glossary_name_check_min_chars', 'glossary_name_check_autofix_level'):\n"
  "            if hasattr(self, key + '_var'):\n"
  "                config[key] = getattr(self, key + '_var')\n"
  '        return name_check.settings_from_config(config)\n'
  '\n'
  '    def open_glossary_name_fix_dialog(self, parent=None):\n'
  '        """Check glossary names in existing output: review the names that differ from the glossary in\n'
  '        the output folder of every selected input file and fix the checked ones."""\n'
  '        import glossary_name_check as name_check\n'
  '        parent = parent or self\n'
  '        title = "Check Glossary Names"\n'
  '        folders = self._glossary_name_fix_folders(parent)\n'
  '        if not folders:\n'
  '            return\n'
  "        use_ner = bool(getattr(self, 'glossary_name_check_use_ner_var',\n"
  "                               self.config.get('glossary_name_check_use_ner', True)))\n"
  '        if use_ner:\n'
  '            import ner_names\n'
  '            if ner_names.runtime_available() and not ner_names.model_installed():\n'
  '                answer = self._glossary_name_message(\n'
  '                    parent, title,\n'
  '                    "The glossary name check finds names with an English named-entity model (NER, 104 '
  'MB).\\n"\n'
  '                    "Download it now? No uses the capitalization heuristic this time.",\n'
  '                    QMessageBox.Question, QMessageBox.Yes | QMessageBox.No)\n'
  '                use_ner = answer == QMessageBox.Yes\n'
  '        try:\n'
  '            reviews = self._glossary_name_review_with_progress(parent, title, folders, use_ner)\n'
  '        except Exception as e:\n'
  '            self._glossary_name_message(parent, title, f"The glossary name check failed:\\n{e}", '
  'QMessageBox.Warning)\n'
  '            return\n'
  '        if reviews is None:\n'
  '            return\n'
  '        if not any(r.chapters for r in reviews):\n'
  '            lines = [r.problem or name_check.ALL_MATCH for r in reviews]\n'
  '            if len(reviews) > 1:\n'
  '                lines = [f"{os.path.basename(r.folder)}: {line}" for r, line in zip(reviews, lines)]\n'
  '            self._glossary_name_message(parent, title, "\\n\\n".join(lines))\n'
  '            return\n'
  '\n'
  '        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QHeaderView, QStackedWidget\n'
  '        from qt_widget_helpers import CheckmarkDelegate, NavComboRow\n'
  '        review = QDialog(parent)\n'
  '        review.setWindowTitle(title)\n'
  '        review.resize(960, 600)\n'
  '        layout = QVBoxLayout(review)\n'
  '        layout.setContentsMargins(18, 16, 18, 14)\n'
  '        layout.setSpacing(10)\n'
  '        # one page per output folder; the ◀ folder ▶ switcher (as in the glossary editor) when several\n'
  '        folder_row = NavComboRow("Output folder:", "Previous folder", "Next folder")\n'
  '        pages = QStackedWidget()\n'
  '        rows = []  # (item, folder, output file, finding)\n'
  '        for folder_review in reviews:\n'
  '            page = QWidget()\n'
  '            page_layout = QVBoxLayout(page)\n'
  '            page_layout.setContentsMargins(0, 0, 0, 0)\n'
  '            page_layout.setSpacing(10)\n'
  '            chapters = folder_review.chapters\n'
  '            total = sum(len(c.reportable) for c in chapters)\n'
  '            if chapters:\n'
  '                text = (f"{total} name(s) in {len(chapters)} file(s) differ from "\n'
  '                        f"<b>{_html.escape(os.path.basename(folder_review.glossary_path))}</b>.<br>"\n'
  '                        "Checked rows are replaced with the glossary name. Rows are pre-checked when one '
  'close "\n'
  '                        "spelling of a character name was found; review the others before checking '
  'them.<br>"\n'
  '                        "Each changed file is backed up to <i>glossary_name_check_backup</i> first.")\n'
  '                label = f"{os.path.basename(folder_review.folder)} — {total} name(s)"\n'
  '            else:\n'
  '                text = _html.escape(folder_review.problem or name_check.ALL_MATCH)\n'
  '                label = f"{os.path.basename(folder_review.folder)} — {\'not checked\' if '
  'folder_review.problem else \'all match\'}"\n'
  '            intro = QLabel(text)\n'
  '            intro.setTextFormat(Qt.RichText)\n'
  '            intro.setWordWrap(True)\n'
  '            page_layout.addWidget(intro)\n'
  '            tree = QTreeWidget()\n'
  '            tree.setColumnCount(5)\n'
  '            tree.setHeaderLabels(["Fix", "Chapter", "Found in output", "Glossary name", "Times"])\n'
  '            tree.setRootIsDecorated(False)\n'
  '            tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)\n'
  "            # the app's checkbox (Other Settings / EPUB Library) instead of Qt's bare tick\n"
  '            tree.setItemDelegateForColumn(0, CheckmarkDelegate(tree))\n'
  '            tree.setContextMenuPolicy(Qt.CustomContextMenu)\n'
  '            tree.customContextMenuRequested.connect(\n'
  '                lambda pos, t=tree: self._glossary_name_row_menu(review, t, pos, rows))\n'
  '            for chapter in chapters:\n'
  '                for finding in chapter.reportable:\n'
  '                    item = QTreeWidgetItem(["", chapter.output_file, " / ".join(finding.variants),\n'
  '                                            finding.expected, str(finding.occurrences)])\n'
  '                    item.setCheckState(0, Qt.Checked if finding.fixable else Qt.Unchecked)\n'
  '                    item.setToolTip(2, f"Source: {finding.raw} ({finding.entry_type})")\n'
  '                    item.setToolTip(3, f"From: {finding.origin or \'Book glossary\'}")\n'
  '                    tree.addTopLevelItem(item)\n'
  '                    rows.append((item, folder_review.folder, chapter.output_file, finding))\n'
  '            tree.setVisible(bool(chapters))\n'
  '            page_layout.addWidget(tree, 1)\n'
  '            pages.addWidget(page)\n'
  '            folder_row.combo.addItem(label, folder_review.folder)\n'
  '            folder_row.combo.setItemData(folder_row.combo.count() - 1, folder_review.folder, '
  'Qt.ToolTipRole)\n'
  '        folder_row.combo.currentIndexChanged.connect(pages.setCurrentIndex)\n'
  '        folder_row.update_arrows()\n'
  '        folder_row.setVisible(len(reviews) > 1)\n'
  '        first = next((i for i, r in enumerate(reviews) if r.chapters), 0)\n'
  '        folder_row.combo.setCurrentIndex(first)\n'
  '        pages.setCurrentIndex(first)\n'
  '        layout.addWidget(folder_row)\n'
  '        layout.addWidget(pages, 1)\n'
  '\n'
  '        buttons = QHBoxLayout()\n'
  '        check_all = QPushButton("Check all")\n'
  '        uncheck_all = QPushButton("Uncheck all")\n'
  '        apply_btn = QPushButton("Apply checked")\n'
  '        apply_btn.setStyleSheet("background-color: #28a745; color: white; padding: 4px 12px;")\n'
  '        cancel_btn = QPushButton("Cancel")\n'
  '        for button in (check_all, uncheck_all):\n'
  '            buttons.addWidget(button)\n'
  '        buttons.addStretch()\n'
  '        buttons.addWidget(apply_btn)\n'
  '        buttons.addWidget(cancel_btn)\n'
  '        layout.addLayout(buttons)\n'
  '\n'
  '        def _set_shown(state):\n'
  '            # Check all / Uncheck all act on the folder shown\n'
  '            folder = folder_row.combo.currentData()\n'
  '            for item, row_folder, _output_file, _finding in rows:\n'
  '                if row_folder == folder:\n'
  '                    item.setCheckState(0, state)\n'
  '\n'
  '        check_all.clicked.connect(lambda: _set_shown(Qt.Checked))\n'
  '        uncheck_all.clicked.connect(lambda: _set_shown(Qt.Unchecked))\n'
  '        cancel_btn.clicked.connect(review.reject)\n'
  '\n'
  '        def _apply():\n'
  '            plans = {}\n'
  '            for item, folder, output_file, finding in rows:\n'
  '                if item.checkState(0) == Qt.Checked:\n'
  '                    plans.setdefault(folder, {}).setdefault(output_file, []).append(finding)\n'
  '            if not plans:\n'
  '                self._glossary_name_message(review, title, "No rows are checked.")\n'
  '                return\n'
  '            files = sum(len(plan) for plan in plans.values())\n'
  '            where = f" in {len(plans)} output folders" if len(plans) > 1 else ""\n'
  '            confirm = self._glossary_name_message(\n'
  '                review, title,\n'
  '                f"Replace the checked names in {files} file(s){where}?\\n"\n'
  '                "The original files are backed up to glossary_name_check_backup.",\n'
  '                QMessageBox.Question, QMessageBox.Yes | QMessageBox.No)\n'
  '            if confirm != QMessageBox.Yes:\n'
  '                return\n'
  '            replaced = changed_files = 0\n'
  '            for folder, plan in plans.items():\n'
  '                changed = name_check.fix_files(folder, plan)\n'
  '                replaced += sum(changed.values())\n'
  '                changed_files += len(changed)\n'
  "                if hasattr(self, 'append_log'):\n"
  '                    self.append_log(f"📖 Glossary name fix: {sum(changed.values())} replacement(s) in "\n'
  '                                    f"{len(changed)} file(s) in {folder}")\n'
  '            self._glossary_name_message(\n'
  '                review, title,\n'
  '                f"Replaced {replaced} name(s) in {changed_files} file(s).\\n"\n'
  '                "Run a QA scan again to clear these chapters\' QA issues.")\n'
  '            review.accept()\n'
  '\n'
  '        apply_btn.clicked.connect(_apply)\n'
  '        review.exec()\n'
  '\n'
  '    def prewarm_qa_scanner_gui(self):\n'),
 ("                    'check_glossary_leakage': True,\n                    'min_file_length': 0,\n",
  "                    'check_glossary_leakage': True,\n"
  "                    'check_glossary_names': False,\n"
  "                    'min_file_length': 0,\n"),
 ('            return False\n\n',
  '            return False\n'
  '\n'
  '    def _qa_auto_search_sources(self):\n'
  '        """Source files whose output folders auto-search looks for: the current selection, else the\n'
  '        stored source file. Returns ``(sources, stored_source, direct_text_only)``; Direct Text sources\n'
  '        are left out (``direct_text_only`` when nothing else was selected)."""\n'
  '        epub_files_to_scan = []\n'
  '        primary_epub_path = None\n'
  '        # First check if current selection contains a supported source file.\n'
  '        current_epub_files = []\n'
  "        if hasattr(self, 'selected_files') and self.selected_files:\n"
  '            current_epub_files = [\n'
  '                f for f in self.selected_files\n'
  "                if f.lower().endswith(('.epub', '.txt', '.pdf', '.md', '.html', '.htm', '.xhtml'))\n"
  '            ]\n'
  "            epub_count = len([f for f in current_epub_files if f.lower().endswith('.epub')])\n"
  "            txt_count = len([f for f in current_epub_files if f.lower().endswith('.txt')])\n"
  "            pdf_count = len([f for f in current_epub_files if f.lower().endswith('.pdf')])\n"
  "            md_count = len([f for f in current_epub_files if f.lower().endswith('.md')])\n"
  '            html_count = len([\n'
  '                f for f in current_epub_files\n'
  "                if f.lower().endswith(('.html', '.htm', '.xhtml'))\n"
  '            ])\n'
  '            print(\n'
  '                f"[DEBUG] Current selection contains {epub_count} EPUB files, "\n'
  '                f"{txt_count} TXT files, {pdf_count} PDF files, {md_count} MD files, "\n'
  '                f"and {html_count} HTML files"\n'
  '            )\n'
  '\n'
  '        if current_epub_files:\n'
  '            # Use source files from current selection\n'
  '            epub_files_to_scan = current_epub_files\n'
  '            print(f"[DEBUG] Using {len(epub_files_to_scan)} source files from current selection")\n'
  '        else:\n'
  '            # No source files in current selection - check if we have stored path\n'
  '            primary_epub_path = self.get_current_epub_path()\n'
  '            print(f"[DEBUG] get_current_epub_path returned: {primary_epub_path}")\n'
  '\n'
  '            if primary_epub_path:\n'
  '                epub_files_to_scan = [primary_epub_path]\n'
  '                print(f"[DEBUG] Using stored source file for auto-search")\n'
  '\n'
  '        # Direct Text owns a separate temporary/persistent output workflow and\n'
  '        # must never seed automatic QA source or output discovery.\n'
  '        try:\n'
  '            from qa_scan_runtime import is_direct_text_qa_path\n'
  '            direct_text_sources = [\n'
  '                source_path for source_path in epub_files_to_scan\n'
  '                if is_direct_text_qa_path(source_path)\n'
  '            ]\n'
  '            epub_files_to_scan = [\n'
  '                source_path for source_path in epub_files_to_scan\n'
  '                if not is_direct_text_qa_path(source_path)\n'
  '            ]\n'
  '            for skipped_source in direct_text_sources:\n'
  '                self.append_log(f"⏭️ Excluding Direct Text source from QA scan: {skipped_source}")\n'
  '            if direct_text_sources and not epub_files_to_scan:\n'
  '                self.append_log("⏭️ QA scan skipped: Direct Text cannot trigger automatic QA scanning.")\n'
  '                return epub_files_to_scan, primary_epub_path, True\n'
  '        except Exception:\n'
  '            pass\n'
  '        return epub_files_to_scan, primary_epub_path, False\n'
  '\n'
  '    def _qa_auto_search_enabled(self):\n'
  '        """The QA scanner\'s Auto-search output toggle (checkbox, else saved config)."""\n'
  "        auto_search_enabled = self.config.get('qa_auto_search_output', True)\n"
  "        if hasattr(self, 'qa_auto_search_output_checkbox'):\n"
  '            try:\n'
  '                auto_search_enabled = bool(self.qa_auto_search_output_checkbox.isChecked())\n'
  '            except Exception:\n'
  '                pass\n'
  '        return auto_search_enabled\n'
  '\n'
  '    def _auto_search_output_folders(self, epub_files_to_scan):\n'
  '        """Auto-search output: the output folder of each source file that has translated files."""\n'
  '        folders_to_scan = []\n'
  '        self.append_log(f"🔍 DEBUG: Auto-search running with {len(epub_files_to_scan)} EPUB files")\n'
  '        for epub_path in epub_files_to_scan:\n'
  '            self.append_log(f"🔍 DEBUG: Processing EPUB: {epub_path}")\n'
  '            try:\n'
  '                epub_base = os.path.splitext(os.path.basename(epub_path))[0]\n'
  "                # Use EPUB's parent dir as the primary base (not os.getcwd(), which\n"
  '                # returns "/" on macOS frozen .app bundles)\n'
  '                epub_parent_dir = os.path.dirname(os.path.abspath(epub_path))\n'
  '                current_dir = os.getcwd()\n'
  '                script_dir = os.path.dirname(os.path.abspath(__file__))\n'
  '\n'
  '                self.append_log(f"🔍 DEBUG: EPUB base name: \'{epub_base}\'")\n'
  '                self.append_log(f"🔍 DEBUG: EPUB parent dir: {epub_parent_dir}")\n'
  '                self.append_log(f"🔍 DEBUG: Current dir: {current_dir}")\n'
  '                self.append_log(f"🔍 DEBUG: Script dir: {script_dir}")\n'
  '\n'
  "                override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')\n"
  '                if override_dir:\n'
  '                    self.append_log(f"🔍 DEBUG: Checking override dir: {override_dir}")\n'
  '\n'
  '                from qa_scan_runtime import automatic_qa_output_candidates\n'
  '                candidates = automatic_qa_output_candidates(\n'
  '                    epub_path,\n'
  '                    current_dir=current_dir,\n'
  '                    script_dir=script_dir,\n'
  '                    output_root=override_dir,\n'
  '                )\n'
  '\n'
  '                folder_found = None\n'
  '                for i, candidate in enumerate(candidates):\n'
  '                    exists = os.path.isdir(candidate)\n'
  '                    self.append_log(f"  [{epub_base}] Checking candidate {i+1}: {candidate} - {\'EXISTS\' '
  'if exists else \'NOT FOUND\'}")\n'
  '\n'
  '                    if exists:\n'
  '                        # Verify the folder actually contains appropriate files (HTML/XHTML or TXT)\n'
  '                        try:\n'
  '                            files = os.listdir(candidate)\n'
  '                            # Determine if text file mode is enabled\n'
  "                            text_file_mode = self.config.get('qa_text_file_mode', False)\n"
  "                            if hasattr(self, 'qa_text_file_mode_checkbox'):\n"
  '                                try:\n'
  '                                    text_file_mode = bool(self.qa_text_file_mode_checkbox.isChecked())\n'
  '                                except Exception:\n'
  '                                    pass\n'
  '\n'
  '                            # Auto-detect text file mode if source file is .txt or .pdf\n'
  "                            if epub_path and epub_path.lower().endswith(('.txt', '.pdf')):\n"
  '                                text_file_mode = True\n'
  '\n'
  '                            if text_file_mode:\n'
  '                                # For text mode, check for both .txt AND .html files (PDFs generate '
  '.html)\n'
  '                                target_files = [\n'
  '                                    f for f in files\n'
  "                                    if f.lower().endswith('.txt') or is_html_like_path(f)\n"
  '                                ]\n'
  '                                file_type = "TXT/HTML"\n'
  '                            else:\n'
  '                                target_files = [f for f in files if is_html_like_path(f)]\n'
  '                                file_type = "HTML/XHTML"\n'
  '\n'
  '                            if target_files:\n'
  '                                folder_found = candidate\n'
  '                                self.append_log(f"📁 Auto-selected output folder for {epub_base}: '
  '{folder_found}")\n'
  '                                self.append_log(f"   Found {len(target_files)} {file_type} files to '
  'scan")\n'
  '                                break\n'
  '                            else:\n'
  '                                self.append_log(f"  [{epub_base}] Folder exists but contains no '
  '{file_type} files: {candidate}")\n'
  '                        except Exception as e:\n'
  '                            self.append_log(f"  [{epub_base}] Error checking files in {candidate}: {e}")\n'
  '\n'
  '                if folder_found:\n'
  '                    folders_to_scan.append(folder_found)\n'
  '                    self.append_log(f"🔍 DEBUG: Added to folders_to_scan: {folder_found}")\n'
  '                else:\n'
  '                    self.append_log(f"  ⚠️ No output folder found for {epub_base}")\n'
  '\n'
  '            except Exception as e:\n'
  '                self.append_log(f"  ❌ Error processing {epub_base}: {e}")\n'
  '\n'
  '        self.append_log(f"🔍 DEBUG: Final folders_to_scan: {folders_to_scan}")\n'
  '        return folders_to_scan\n'
  '\n'),
 ('        # ALWAYS populate epub_files_to_scan for auto-search, regardless of word count checking\n'
  '        # First check if current selection contains a supported source file.\n'
  '        current_epub_files = []\n'
  "        if hasattr(self, 'selected_files') and self.selected_files:\n"
  '            current_epub_files = [\n'
  '                f for f in self.selected_files\n'
  "                if f.lower().endswith(('.epub', '.txt', '.pdf', '.md', '.html', '.htm', '.xhtml'))\n"
  '            ]\n'
  "            epub_count = len([f for f in current_epub_files if f.lower().endswith('.epub')])\n"
  "            txt_count = len([f for f in current_epub_files if f.lower().endswith('.txt')])\n"
  "            pdf_count = len([f for f in current_epub_files if f.lower().endswith('.pdf')])\n"
  "            md_count = len([f for f in current_epub_files if f.lower().endswith('.md')])\n"
  '            html_count = len([\n'
  '                f for f in current_epub_files\n'
  "                if f.lower().endswith(('.html', '.htm', '.xhtml'))\n"
  '            ])\n'
  '            print(\n'
  '                f"[DEBUG] Current selection contains {epub_count} EPUB files, "\n'
  '                f"{txt_count} TXT files, {pdf_count} PDF files, {md_count} MD files, "\n'
  '                f"and {html_count} HTML files"\n'
  '            )\n'
  '\n'
  '        if current_epub_files:\n'
  '            # Use source files from current selection\n'
  '            epub_files_to_scan = current_epub_files\n'
  '            print(f"[DEBUG] Using {len(epub_files_to_scan)} source files from current selection")\n'
  '        else:\n'
  '            # No source files in current selection - check if we have stored path\n'
  '            primary_epub_path = self.get_current_epub_path()\n'
  '            print(f"[DEBUG] get_current_epub_path returned: {primary_epub_path}")\n'
  '\n'
  '            if primary_epub_path:\n'
  '                epub_files_to_scan = [primary_epub_path]\n'
  '                print(f"[DEBUG] Using stored source file for auto-search")\n'
  '\n'
  '        # Direct Text owns a separate temporary/persistent output workflow and\n'
  '        # must never seed automatic QA source or output discovery.\n'
  '        try:\n'
  '            from qa_scan_runtime import is_direct_text_qa_path\n'
  '            direct_text_sources = [\n'
  '                source_path for source_path in epub_files_to_scan\n'
  '                if is_direct_text_qa_path(source_path)\n'
  '            ]\n'
  '            epub_files_to_scan = [\n'
  '                source_path for source_path in epub_files_to_scan\n'
  '                if not is_direct_text_qa_path(source_path)\n'
  '            ]\n'
  '            for skipped_source in direct_text_sources:\n'
  '                self.append_log(f"⏭️ Excluding Direct Text source from QA scan: {skipped_source}")\n'
  '            if direct_text_sources and not epub_files_to_scan:\n'
  '                self.append_log("⏭️ QA scan skipped: Direct Text cannot trigger automatic QA scanning.")\n'
  '                return\n'
  '        except Exception:\n'
  '            pass\n'
  '\n',
  '        # ALWAYS populate epub_files_to_scan for auto-search, regardless of word count checking\n'
  '        epub_files_to_scan, primary_epub_path, direct_text_only = self._qa_auto_search_sources()\n'
  '        if direct_text_only:\n'
  '            return\n'
  '\n'),
 ('        # Get auto-search preference from checkbox if it exists, otherwise from config\n'
  "        auto_search_enabled = self.config.get('qa_auto_search_output', True)\n"
  "        if hasattr(self, 'qa_auto_search_output_checkbox'):\n"
  '            try:\n'
  '                auto_search_enabled = bool(self.qa_auto_search_output_checkbox.isChecked())\n'
  '            except Exception:\n'
  '                pass\n'
  '\n'
  '        # Debug output for scanning phase removed\n'
  '\n',
  '        # Get auto-search preference from checkbox if it exists, otherwise from config\n'
  '        auto_search_enabled = self._qa_auto_search_enabled()\n'
  '\n'),
 ('            # Process each EPUB file to find its corresponding output folder\n'
  '            self.append_log(f"🔍 DEBUG: Auto-search running with {len(epub_files_to_scan)} EPUB files")\n'
  '            for epub_path in epub_files_to_scan:\n'
  '                self.append_log(f"🔍 DEBUG: Processing EPUB: {epub_path}")\n'
  '                try:\n'
  '                    epub_base = os.path.splitext(os.path.basename(epub_path))[0]\n'
  "                    # Use EPUB's parent dir as the primary base (not os.getcwd(), which\n"
  '                    # returns "/" on macOS frozen .app bundles)\n'
  '                    epub_parent_dir = os.path.dirname(os.path.abspath(epub_path))\n'
  '                    current_dir = os.getcwd()\n'
  '                    script_dir = os.path.dirname(os.path.abspath(__file__))\n'
  '\n'
  '                    self.append_log(f"🔍 DEBUG: EPUB base name: \'{epub_base}\'")\n'
  '                    self.append_log(f"🔍 DEBUG: EPUB parent dir: {epub_parent_dir}")\n'
  '                    self.append_log(f"🔍 DEBUG: Current dir: {current_dir}")\n'
  '                    self.append_log(f"🔍 DEBUG: Script dir: {script_dir}")\n'
  '\n'
  "                    override_dir = os.environ.get('OUTPUT_DIRECTORY') or "
  "self.config.get('output_directory')\n"
  '                    if override_dir:\n'
  '                        self.append_log(f"🔍 DEBUG: Checking override dir: {override_dir}")\n'
  '\n'
  '                    from qa_scan_runtime import automatic_qa_output_candidates\n'
  '                    candidates = automatic_qa_output_candidates(\n'
  '                        epub_path,\n'
  '                        current_dir=current_dir,\n'
  '                        script_dir=script_dir,\n'
  '                        output_root=override_dir,\n'
  '                    )\n'
  '\n'
  '                    folder_found = None\n'
  '                    for i, candidate in enumerate(candidates):\n'
  '                        exists = os.path.isdir(candidate)\n'
  '                        self.append_log(f"  [{epub_base}] Checking candidate {i+1}: {candidate} - '
  '{\'EXISTS\' if exists else \'NOT FOUND\'}")\n'
  '\n'
  '                        if exists:\n'
  '                            # Verify the folder actually contains appropriate files (HTML/XHTML or TXT)\n'
  '                            try:\n'
  '                                files = os.listdir(candidate)\n'
  '                                # Determine if text file mode is enabled\n'
  "                                text_file_mode = self.config.get('qa_text_file_mode', False)\n"
  "                                if hasattr(self, 'qa_text_file_mode_checkbox'):\n"
  '                                    try:\n'
  '                                        text_file_mode = '
  'bool(self.qa_text_file_mode_checkbox.isChecked())\n'
  '                                    except Exception:\n'
  '                                        pass\n'
  '\n'
  '                                # Auto-detect text file mode if source file is .txt or .pdf\n'
  "                                if epub_path and epub_path.lower().endswith(('.txt', '.pdf')):\n"
  '                                    text_file_mode = True\n'
  '\n'
  '                                if text_file_mode:\n'
  '                                    # For text mode, check for both .txt AND .html files (PDFs generate '
  '.html)\n'
  '                                    target_files = [\n'
  '                                        f for f in files\n'
  "                                        if f.lower().endswith('.txt') or is_html_like_path(f)\n"
  '                                    ]\n'
  '                                    file_type = "TXT/HTML"\n'
  '                                else:\n'
  '                                    target_files = [f for f in files if is_html_like_path(f)]\n'
  '                                    file_type = "HTML/XHTML"\n'
  '\n'
  '                                if target_files:\n'
  '                                    folder_found = candidate\n'
  '                                    self.append_log(f"📁 Auto-selected output folder for {epub_base}: '
  '{folder_found}")\n'
  '                                    self.append_log(f"   Found {len(target_files)} {file_type} files to '
  'scan")\n'
  '                                    break\n'
  '                                else:\n'
  '                                    self.append_log(f"  [{epub_base}] Folder exists but contains no '
  '{file_type} files: {candidate}")\n'
  '                            except Exception as e:\n'
  '                                self.append_log(f"  [{epub_base}] Error checking files in {candidate}: '
  '{e}")\n'
  '\n'
  '                    if folder_found:\n'
  '                        folders_to_scan.append(folder_found)\n'
  '                        self.append_log(f"🔍 DEBUG: Added to folders_to_scan: {folder_found}")\n'
  '                    else:\n'
  '                        self.append_log(f"  ⚠️ No output folder found for {epub_base}")\n'
  '\n'
  '                except Exception as e:\n'
  '                    self.append_log(f"  ❌ Error processing {epub_base}: {e}")\n'
  '\n'
  '            self.append_log(f"🔍 DEBUG: Final folders_to_scan: {folders_to_scan}")\n'
  '\n',
  '            # Process each EPUB file to find its corresponding output folder\n'
  '            folders_to_scan.extend(self._auto_search_output_folders(epub_files_to_scan))\n'
  '\n'),
 ('            detection_layout.addWidget(check_glossary_checkbox)\n\n',
  '            detection_layout.addWidget(check_glossary_checkbox)\n'
  '\n'
  '            check_glossary_names_checkbox = self._create_styled_checkbox(\n'
  '                "Check glossary names with NER (Experimental)")\n'
  "            check_glossary_names_checkbox.setChecked(qa_settings.get('check_glossary_names', False))\n"
  '            check_glossary_names_checkbox.setToolTip(\n'
  '                "Flags chapters where a glossary name from the source chapter was translated with a\\n"\n'
  '                "different spelling (e.g. \'Walter Argent\' for glossary \'Valter Argent\').\\n"\n'
  '                "Names are found with a named-entity recognition model (NER, 104 MB, downloaded on\\n"\n'
  '                "first use) plus a capitalization heuristic; Other Settings › \\"Use Named Entity\\n"\n'
  '                "Recognition model (NER)\\" turns NER off (builds without ONNX Runtime use the '
  'heuristic).\\n"\n'
  '                "Uses the entry types of Emergency Glossary Compliance and the name check\'s minimum\\n"\n'
  '                "length (Other Settings). Flagged chapters count as QA failures like other issues.")\n'
  '            detection_layout.addWidget(check_glossary_names_checkbox)\n'
  '\n'
  '            glossary_names_fix_row = QWidget()\n'
  '            glossary_names_fix_layout = QHBoxLayout(glossary_names_fix_row)\n'
  '            glossary_names_fix_layout.setContentsMargins(20, 0, 0, 5)\n'
  '            glossary_names_fix_btn = QPushButton("Check glossary names in existing output…")\n'
  '            glossary_names_fix_btn.setStyleSheet(\n'
  '                "background-color: #3a5f8a; color: white; padding: 4px 8px; border-radius: 3px;")\n'
  '            glossary_names_fix_btn.setToolTip(\n'
  '                "Scan the output folder of every selected input file (Auto-search output; else pick '
  'one),\\n"\n'
  '                "including chapter titles, the TOC and the book metadata, review every name that '
  'differs\\n"\n'
  '                "from the glossary and replace the ones you keep checked. Changed files are backed up '
  'first.")\n'
  '            glossary_names_fix_btn.clicked.connect(\n'
  '                lambda: self.open_glossary_name_fix_dialog(dialog))\n'
  '            glossary_names_fix_layout.addWidget(glossary_names_fix_btn)\n'
  '            glossary_names_fix_layout.addStretch()\n'
  '            detection_layout.addWidget(glossary_names_fix_row)\n'
  '            glossary_names_fix_row.setVisible(check_glossary_names_checkbox.isChecked())\n'
  '            check_glossary_names_checkbox.toggled.connect(glossary_names_fix_row.setVisible)\n'
  '\n'),
 ("                        'check_glossary_leakage': (check_glossary_checkbox, lambda x: x.isChecked()),\n"
  "                        'check_potential_truncation': (check_potential_truncation_checkbox, lambda x: "
  'x.isChecked()),\n',
  "                        'check_glossary_leakage': (check_glossary_checkbox, lambda x: x.isChecked()),\n"
  "                        'check_glossary_names': (check_glossary_names_checkbox, lambda x: "
  'x.isChecked()),\n'
  "                        'check_potential_truncation': (check_potential_truncation_checkbox, lambda x: "
  'x.isChecked()),\n'),
 ("                            ('QA_CHECK_GLOSSARY_LEAKAGE', '1' if "
  "qa_settings.get('check_glossary_leakage', True) else '0'),\n"
  "                            ('QA_CHECK_MISSING_IMAGES', '1' if qa_settings.get('check_missing_images', "
  "True) else '0'),\n",
  "                            ('QA_CHECK_GLOSSARY_LEAKAGE', '1' if "
  "qa_settings.get('check_glossary_leakage', True) else '0'),\n"
  "                            ('QA_CHECK_GLOSSARY_NAMES', '1' if qa_settings.get('check_glossary_names', "
  "False) else '0'),\n"
  "                            ('QA_CHECK_MISSING_IMAGES', '1' if qa_settings.get('check_missing_images', "
  "True) else '0'),\n"),
 ("                    ('check_glossary_leakage', check_glossary_checkbox, True),\n"
  "                    ('check_potential_truncation', check_potential_truncation_checkbox, False),\n",
  "                    ('check_glossary_leakage', check_glossary_checkbox, True),\n"
  "                    ('check_glossary_names', check_glossary_names_checkbox, False),\n"
  "                    ('check_potential_truncation', check_potential_truncation_checkbox, False),\n")]
_NAME_CHECK_QA_RUNTIME = [('        "check_glossary_leakage": True,\n        "check_potential_truncation": False,\n',
  '        "check_glossary_leakage": True,\n'
  '        "check_glossary_names": False,\n'
  '        "check_potential_truncation": False,\n'),
 ('        "QA_CHECK_GLOSSARY_LEAKAGE": "1" if settings.get("check_glossary_leakage", True) else "0",\n'
  '        "QA_CHECK_MISSING_IMAGES": "1" if settings.get("check_missing_images", True) else "0",\n',
  '        "QA_CHECK_GLOSSARY_LEAKAGE": "1" if settings.get("check_glossary_leakage", True) else "0",\n'
  '        "QA_CHECK_GLOSSARY_NAMES": "1" if settings.get("check_glossary_names", False) else "0",\n'
  '        "QA_CHECK_MISSING_IMAGES": "1" if settings.get("check_missing_images", True) else "0",\n'),
 ('        )\n'
  '    )\n'
  '\n'
  '    output_language = live_config.get("output_language") or live_config.get("output_language_var")\n',
  '        )\n'
  '    )\n'
  '\n'
  '    if settings.get("check_glossary_names"):\n'
  "        # Glossary name check: Emergency Glossary Compliance's entry types, its own minimum length / NER\n"
  '        try:\n'
  '            from glossary_name_check import layer_settings_from_config, settings_from_config, '
  'use_ner_from_config\n'
  '            settings["_glossary_name_scope"] = settings_from_config(live_config)\n'
  '            settings["_glossary_name_use_ner"] = use_ner_from_config(live_config)\n'
  '            # the glossaries a request adds (Add Additional Glossary / Enable Unified Glossary)\n'
  '            settings["_glossary_name_layers"] = layer_settings_from_config(live_config)\n'
  '        except Exception:\n'
  '            pass\n'
  '\n'
  '    output_language = live_config.get("output_language") or live_config.get("output_language_var")\n')]
_NAME_CHECK_SCANNER = [('\ndef check_potential_truncation(raw_text):\n',
  '\n'
  'def run_glossary_name_check(folder_path, results, qa_settings, log=print, *, epub_path=None,\n'
  '                            progress_path=None, should_stop=None):\n'
  '    """Experimental: flag translated names that drifted from the glossary (glossary_name_check)."""\n'
  '    import glossary_name_check as name_check\n'
  '    glossary_path = name_check.find_glossary_for_folder(folder_path, epub_path)\n'
  '    if not glossary_path:\n'
  '        log("   ⏭️ Glossary name check: no glossary found for this folder")\n'
  '        return 0\n'
  '    scope = qa_settings.get("_glossary_name_scope") or name_check.settings_from_env()\n'
  '    layer_settings = qa_settings.get("_glossary_name_layers")  # None: the translation env\n'
  '    entries = name_check.load_entries(glossary_path, settings=layer_settings, **scope)\n'
  '    if not entries:\n'
  '        log("   ⏭️ Glossary name check: no glossary entries match the Emergency Glossary Compliance entry '
  'types")\n'
  '        return 0\n'
  "    by_file = {r.get('filename'): r for r in results if isinstance(r, dict) and r.get('filename')}\n"
  '    sources = " + ".join([os.path.basename(glossary_path)] + [\n'
  '        origin for _path, origin in name_check.glossary_layers(glossary_path, layer_settings)[1:]])\n'
  '    log(f"\\n📖 Glossary name check (experimental): {len(entries.entries)} entries from {sources}")\n'
  '    use_ner = qa_settings.get("_glossary_name_use_ner")\n'
  '    ner = name_check.resolve_ner(name_check.use_ner_from_env() if use_ner is None else use_ner, log)\n'
  '    chapters = name_check.scan_folder(folder_path, entries, progress_path=progress_path,\n'
  '                                      selected_files=list(by_file), stop=should_stop, ner=ner)\n'
  '    flagged = 0\n'
  '    for chapter in chapters:\n'
  '        result_obj = by_file.get(chapter.output_file)\n'
  '        if result_obj is None:\n'
  '            continue\n'
  "        issues = result_obj.setdefault('issues', [])\n"
  '        for finding in chapter.reportable:\n'
  '            label = "glossary_name_variant" if finding.fixable else "glossary_name_mismatch"\n'
  '            issue_code = f"{label} ({finding.describe()})"\n'
  '            if issue_code not in issues:\n'
  '                issues.append(issue_code)\n'
  "        result_obj['score'] = len(issues)\n"
  '        flagged += 1\n'
  '    log(f"   ✅ Glossary name check: {flagged} file(s) with names that differ from the glossary")\n'
  '    return flagged\n'
  '\n'
  '\n'
  'def check_potential_truncation(raw_text):\n'),
 ("            'check_glossary_leakage': True,\n            'check_missing_images': True,\n",
  "            'check_glossary_leakage': True,\n"
  "            'check_glossary_names': False,\n"
  "            'check_missing_images': True,\n"),
 ('    log(f"   ✓ Glossary leakage check: {\'ENABLED\' if qa_settings.get(\'check_glossary_leakage\', True) '
  'else \'DISABLED\'}")\n'
  '    log(f"   ✓ Source resource preservation (images/links/tables/graphics): {\'ENABLED\' if '
  'qa_settings.get(\'check_missing_images\', True) else \'DISABLED\'}")\n',
  '    log(f"   ✓ Glossary leakage check: {\'ENABLED\' if qa_settings.get(\'check_glossary_leakage\', True) '
  'else \'DISABLED\'}")\n'
  '    log(f"   ✓ Glossary name check (experimental): {\'ENABLED\' if '
  'qa_settings.get(\'check_glossary_names\', False) else \'DISABLED\'}")\n'
  '    log(f"   ✓ Source resource preservation (images/links/tables/graphics): {\'ENABLED\' if '
  'qa_settings.get(\'check_missing_images\', True) else \'DISABLED\'}")\n'),
 ('\n    _attach_chunk_results_to_scan(\n',
  '\n'
  "    if qa_settings.get('check_glossary_names', False) and not should_stop():\n"
  '        try:\n'
  '            run_glossary_name_check(folder_path, results, qa_settings, log, epub_path=epub_path,\n'
  '                                    progress_path=progress_path, should_stop=should_stop)\n'
  '        except Exception as e:\n'
  '            log(f"   ⚠️ Glossary name check failed: {e}")\n'
  '\n'
  '    _attach_chunk_results_to_scan(\n')]
_NAME_CHECK_WORKER_QA = _after(
    "            \"check_glossary_leakage\": _env_bool(\"QA_CHECK_GLOSSARY_LEAKAGE\", True),",
    "            \"check_glossary_names\": _env_bool(\"QA_CHECK_GLOSSARY_NAMES\", False),")

# 2026-10-23 Translate's concurrent inpainting (and Clean, through one shared helper) loads an inpainter
# when no preload is filling the pool instead of waiting 60 s; 'Locating' replaces the misleading
# 'Downloading' line printed when the model was already cached
_INPAINT_NO_POOL_SYNC = [('\ndef _run_clean_background(self, image_path: str, regions: list):\n',
  '\n'
  'def _load_inpainter_if_no_pool(self, local_model: str, key) -> bool:\n'
  '    """No pool record for *key* means no preload is filling it (none ran for this model: Glossarion\n'
  '    Mobile, or the model was switched after the startup preload). Load one into the pool now instead of\n'
  '    polling a pool nothing will fill. False when that load failed."""\n'
  '    from manga_translator import MangaTranslator\n'
  '    with MangaTranslator._inpaint_pool_lock:\n'
  '        if key in MangaTranslator._inpaint_pool:\n'
  '            return True\n'
  '    self._log(f"📥 Loading {local_model} inpainter...", "info")\n'
  '    if _load_inpainter_into_pool(self, local_model, key[1], key, False):\n'
  '        return True\n'
  '    self._log(f"❌ Failed to load {local_model} inpainter", "error")\n'
  '    return False\n'
  '\n'
  '\n'
  'def _run_clean_background(self, image_path: str, regions: list):\n'),
 ('                    from local_inpainter import LocalInpainter\n'
  '                    self._log(f"📥 Downloading {local_model} model...", "info")\n'
  '                    temp_inp = LocalInpainter()\n',
  '                    from local_inpainter import LocalInpainter\n'
  '                    self._log(f"🔎 Locating {local_model} model (downloads it only if missing)...", '
  '"info")\n'
  '                    temp_inp = LocalInpainter()\n'),
 ('                        pool_key = (local_model, os.path.abspath(os.path.normpath(resolved_model_path)))\n'
  '                        with MangaTranslator._inpaint_pool_lock:\n'
  '                            has_pool = pool_key in MangaTranslator._inpaint_pool\n'
  '                        if not has_pool:\n'
  '                            self._log(f"📥 Loading {local_model} inpainter...", "info")\n'
  '                            if not _load_inpainter_into_pool(self, local_model, pool_key[1], pool_key, '
  'False):\n'
  '                                self._log(f"❌ Failed to load {local_model} inpainter", "error")\n'
  '                                poll_timeout = 0\n'
  '\n',
  '                        pool_key = (local_model, os.path.abspath(os.path.normpath(resolved_model_path)))\n'
  '                        if not _load_inpainter_if_no_pool(self, local_model, pool_key):\n'
  '                            poll_timeout = 0\n'
  '\n'),
 ('                    from local_inpainter import LocalInpainter\n'
  '                    print(f"[INPAINT_SYNC] Downloading {local_model} model...")\n'
  '                    temp_inp = LocalInpainter()\n',
  '                    from local_inpainter import LocalInpainter\n'
  '                    print(f"[INPAINT_SYNC] Locating {local_model} model (downloads it only if '
  'missing)...")\n'
  '                    temp_inp = LocalInpainter()\n'),
 ('                    poll_timeout = 0 if is_custom_image_edit else 60  # Endpoint-backed edit can '
  'fake-load immediately\n'
  '                    poll_interval = 0.5  # Check every 500ms\n',
  '                    poll_timeout = 0 if is_custom_image_edit else 60  # Endpoint-backed edit can '
  'fake-load immediately\n'
  '                    # No preload filling the pool (e.g. Glossarion Mobile): load one now, do not wait 60 '
  's\n'
  '                    if not is_custom_image_edit and not _load_inpainter_if_no_pool(self, local_model, '
  'key):\n'
  '                        poll_timeout = 0\n'
  '                    poll_interval = 0.5  # Check every 500ms\n'),
 ('            try:\n'
  '                print(f"[PRELOAD_INPAINTER] Downloading {local_model} model for preloading...")\n'
  '                from local_inpainter import LocalInpainter\n',
  '            try:\n'
  '                print(f"[PRELOAD_INPAINTER] Locating {local_model} model for preloading (downloads it '
  'only if missing)...")\n'
  '                from local_inpainter import LocalInpainter\n')]

APPROVED = (
    {
        "date": "2026-10-13",
        "what": "Skip .pdf book title translation is on by default (like .txt titles)",
        "source": {
            "src/owner_state.py": [
                ("('skip_pdf_title_translation_var', 'skip_pdf_title_translation', False)",
                 "('skip_pdf_title_translation_var', 'skip_pdf_title_translation', True)"),
            ],
            "src/translator_gui.py": [
                ("('skip_pdf_title_translation_var', 'skip_pdf_title_translation', False)",
                 "('skip_pdf_title_translation_var', 'skip_pdf_title_translation', True)"),
                ("('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], False, bool)",
                 "('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], True, bool)"),
                # run_env's environment builders lived in translator_gui before U2
                ("getattr(self, 'skip_pdf_title_translation_var', False)",
                 "getattr(self, 'skip_pdf_title_translation_var', True)"),
            ],
            "src/settings_persistence.py": [
                ("('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], False, bool)",
                 "('skip_pdf_title_translation', ['skip_pdf_title_translation_var'], True, bool)"),
            ],
            "src/other_settings.py": [
                ("self.config.get('skip_pdf_title_translation', False)", "self.config.get('skip_pdf_title_translation', True)"),
                ("self.config['skip_pdf_title_translation'] = False", "self.config['skip_pdf_title_translation'] = True"),
            ],
            "src/run_env.py": [
                ("getattr(self, 'skip_pdf_title_translation_var', False)",
                 "getattr(self, 'skip_pdf_title_translation_var', True)"),
            ],
            "src/progress_core.py": [
                ("config.get('skip_pdf_title_translation', False)", "config.get('skip_pdf_title_translation', True)"),
            ],
            "src/TransateKRtoEN.py": [
                ("os.getenv('SKIP_PDF_TITLE_TRANSLATION', '0')", "os.getenv('SKIP_PDF_TITLE_TRANSLATION', '1')"),
            ],
        },
        # recorded values (config key, owner attribute, exported env var): old -> new
        "values": {
            "skip_pdf_title_translation": (False, True),
            "skip_pdf_title_translation_var": (False, True),
            "SKIP_PDF_TITLE_TRANSLATION": ("0", "1"),
        },
    },
    {
        "date": "2026-10-17",
        "what": "New 'Always check and respect Retry-After' toggle (on by default, below Indefinite Rate "
                "Limit Retry); never on together with 'Ignore server Retry-After header'",
        "source": {
            "src/settings_persistence.py": [_RESPECT_SETTINGS_ROW],
            "src/owner_state.py": [_RESPECT_BOOL_ROW, _RESPECT_MIGRATION],
            "src/translator_gui.py": [_RESPECT_SETTINGS_ROW_GUI, _RESPECT_BOOL_ROW, _RESPECT_MIGRATION,
                                      _RESPECT_ENV_DICT, _RESPECT_ENV_PAIR],
            "src/run_env.py": [_RESPECT_ENV_DICT, _RESPECT_ENV_PAIR],
        },
        # a new key: nothing recorded before it existed changes value
        "values": {},
    },
    {
        "date": "2026-10-18",
        "what": "Async batch: Gemini safety settings use each category's threshold (and HARM_CATEGORY_JAILBREAK) "
                "instead of BLOCK_NONE for five categories",
        "source": {
            "src/async_api_processor.py": [_BATCH_SAFETY],
        },
        "values": {},
    },
    {
        "date": "2026-10-18",
        "what": "Per-category Gemini safety thresholds (Other Settings › API Safety Settings); a "
                "category set to 'Same as threshold' follows the main Threshold",
        "source": {
            "src/settings_persistence.py": [_SAFETY_CATEGORY_ROW],
            "src/translator_gui.py": [_SAFETY_CATEGORY_ROW_GUI, _SAFETY_CATEGORY_ENV_DICT, _SAFETY_CATEGORY_ENV_PAIR],
            "src/run_env.py": [_SAFETY_CATEGORY_ENV_DICT, _SAFETY_CATEGORY_ENV_PAIR],
        },
        # a new key: nothing recorded before it existed changes value
        "values": {},
    },
    {
        "date": "2026-10-18",
        "what": "Built-in 'On-device_BeautifulSoup' / 'On-device_html2text' prompt profiles (always included, "
                "cannot be deleted) for small on-device models; the mobile app selects the one matching the text "
                "extraction method when a local/ model is chosen",
        "source": {
            "src/owner_state.py": [_ON_DEVICE_PROFILE_LIST_ROW, _ON_DEVICE_DEFAULT_PROMPT_ROW],
            "src/translator_gui.py": [_ON_DEVICE_PROFILE_LIST_ROW, _ON_DEVICE_DEFAULT_PROMPT_ROW],
        },
        "values": {},
        # recorded profile dicts (under these keys) gain the new profile after the anchor key
        "inserts": [
            {"parents": ("default_prompts", "prompt_profiles"), "after": "Subtitle Translation",
             "key": "On-device_BeautifulSoup", "value": _ON_DEVICE_PROMPT},
            {"parents": ("default_prompts", "prompt_profiles"), "after": "On-device_BeautifulSoup",
             "key": "On-device_html2text", "value": _ON_DEVICE_HTML2TEXT_PROMPT},
        ],
    },
    {
        "date": "2026-10-18",
        "what": "Prompt placeholders stay where the user put them: {split_marker_instruction} is filled "
                "in place, and without an instruction its line is removed without joining its neighbours",
        "source": {
            # the runners lived in translator_gui before U3 / U7
            "src/translator_gui.py": [_SPLIT_MARKER_TEXT_FILE, _SPLIT_MARKER_IMAGE_FILE, _SPLIT_MARKER_RPGMAKER],
            "src/text_jobs.py": [_SPLIT_MARKER_TEXT_FILE],
            "src/image_job.py": [_SPLIT_MARKER_IMAGE_FILE],
            "src/rpgmaker_job.py": [_SPLIT_MARKER_RPGMAKER],
        },
        # prompt text only (the logged preview / length follow from the source change)
        "values": {},
    },
    {
        "date": "2026-10-19",
        "what": "Skip title tag translation is on by default (HTML/XHTML <title> tags are kept untranslated "
                "unless the user turns it off); a saved value still wins",
        "source": {
            "src/owner_state.py": [_TITLE_TAG_CONFIG_GET],
            "src/translator_gui.py": [_TITLE_TAG_CONFIG_GET, _TITLE_TAG_GETATTR, _TITLE_TAG_SETTINGS_ROW,
                                      _USE_TITLE_SETTINGS_ROW],
            "src/settings_persistence.py": [_TITLE_TAG_SETTINGS_ROW, _USE_TITLE_SETTINGS_ROW],
            "src/other_settings.py": [
                _TITLE_TAG_CONFIG_GET,
                ("self.config['skip_title_tag_translation'] = False",
                 "self.config['skip_title_tag_translation'] = True"),
            ],
            "src/run_env.py": [_TITLE_TAG_GETATTR],
            "src/translation_pipeline.py": [_TITLE_TAG_CONFIG_GET],
            "src/app.py": [("config('skip_title_tag_translation', False)", "config('skip_title_tag_translation', True)")],
            "src/qa_scan_runtime.py": [
                ('cfg.get("skip_title_tag_translation", False)', 'cfg.get("skip_title_tag_translation", True)'),
                ('            "SKIP_TITLE_TAG_TRANSLATION",\n            False,\n',
                 '            "SKIP_TITLE_TAG_TRANSLATION",\n            True,\n'),
                ('_env_bool("SKIP_TITLE_TAG_TRANSLATION", False)', '_env_bool("SKIP_TITLE_TAG_TRANSLATION", True)'),
            ],
            "src/Chapter_Extractor.py": [
                ("'skip_title_tag_translation': _env_flag('SKIP_TITLE_TAG_TRANSLATION'),",
                 "'skip_title_tag_translation': _env_flag('SKIP_TITLE_TAG_TRANSLATION', '1'),"),
            ],
            "src/title_tag_translation.py": [
                ("    ``SKIP_TITLE_TAG_TRANSLATION`` is authoritative and defaults to false,\n"
                 "    which means title tags are translated.  ``USE_TITLE`` remains a fallback\n",
                 "    ``SKIP_TITLE_TAG_TRANSLATION`` is authoritative and defaults to true,\n"
                 "    which means title tags are kept untranslated.  ``USE_TITLE`` remains a fallback\n"),
                ('        return _env_true("USE_TITLE")\n    return True\n',
                 '        return _env_true("USE_TITLE")\n    return False\n'),
            ],
        },
        # recorded values (config key, owner attribute, exported env var): old -> new
        "values": {
            "skip_title_tag_translation": (False, True),
            "skip_title_tag_translation_var": (False, True),
            "SKIP_TITLE_TAG_TRANSLATION": ("0", "1"),
            "use_title": (True, False),
            "use_title_var": (True, False),
            "USE_TITLE": ("1", "0"),
        },
    },
    {
        "date": "2026-10-19",
        "what": "New 'Remove excessive empty p and br tags' toggles: On-device (on by default, Glossarion Mobile's "
                "on-device model) and all models (off by default, Other Settings); runs of empty <p>/<br> in the "
                "chapter text sent to the model keep at most two <br/>",
        "source": {
            "src/settings_persistence.py": [_EMPTY_TAGS_SETTINGS_ROW],
            "src/owner_state.py": [_EMPTY_TAGS_BOOL_ROW],
            "src/translator_gui.py": [_EMPTY_TAGS_SETTINGS_ROW_GUI, _EMPTY_TAGS_BOOL_ROW, _EMPTY_TAGS_ENV_DICT,
                                      _EMPTY_TAGS_ENV_PAIR],
            "src/run_env.py": [_EMPTY_TAGS_ENV_DICT, _EMPTY_TAGS_ENV_PAIR],
            "src/async_api_processor.py": [_EMPTY_TAGS_ASYNC_ENV, _EMPTY_TAGS_ASYNC_PAYLOAD],
        },
        # new keys: nothing recorded before they existed changes value
        "values": {},
    },
    {
        "date": "2026-10-19",
        "what": "Time-based API key scheduling: the main key's '⏳ Request Queue' schedule (model name right-click; "
                "{} = off) makes requests wait before sending during its windows (DeepSeek peak hours preset)",
        "source": {
            "src/settings_persistence.py": [_QUEUE_SETTINGS_ROW],
            "src/translator_gui.py": [_QUEUE_SETTINGS_ROW_GUI, _QUEUE_ENV_DICT, _QUEUE_ENV_PAIR],
            "src/run_env.py": [_QUEUE_ENV_DICT, _QUEUE_ENV_PAIR],
        },
        # a new key: nothing recorded before it existed changes value
        "values": {},
    },
    {
        "date": "2026-10-21",
        "what": "RapidOCR reading model on the desktop Manga tab (rapidocr_language: Auto / Japanese-Chinese manga "
                "PP-OCRv6 / Korean PP-OCRv5 / Built-in), shown while RapidOCR is the OCR provider",
        "source": {
            "src/manga_integration.py": [_RAPIDOCR_FRAME, _RAPIDOCR_FRAME_VISIBILITY],
        },
        # an existing key without a desktop control before: nothing recorded changes value
        "values": {},
    },
    {
        "date": "2026-10-21",
        "what": "New on-device OCR provider 'hayai-ocr' (Hayai OCR v2.5 Nova on onnxruntime): the manga tab's "
                "provider combo, its status / Download / Load handling, and per-region crop OCR in the editor",
        "source": {
            "src/manga_integration.py": _HAYAI_MANGA_INTEGRATION,
            "src/ImageRenderer.py": _HAYAI_IMAGERENDERER,
        },
        # a new provider value: nothing recorded before it existed changes value
        "values": {},
    },
    {
        "date": "2026-10-21",
        "what": "API call delay ranges: the API call delay field takes '5' or a range '5-7' (each call waits a "
                "random time between the two); delay_max / SEND_INTERVAL_MAX_SECONDS hold the max; a config "
                "without delay gets 3-7 (was 5)",
        "source": {
            "src/settings_persistence.py": [_DELAY_IMPORT, _DELAY_ROW],
            "src/translator_gui.py": [_DELAY_IMPORT, _DELAY_ROW_GUI, _DELAY_ENV_DICT, _DELAY_FIELD, *_DELAY_RPGMAKER,
                                      _DELAY_GLOSSARY_LOG],
            "src/run_env.py": [_DELAY_IMPORT, _DELAY_ENV_DICT],
            "src/rpgmaker_job.py": [_DELAY_IMPORT, *_DELAY_RPGMAKER],
            "src/translation_pipeline.py": [_DELAY_IMPORT, _DELAY_GLOSSARY_LOG],
            "src/ImageRenderer.py": [_DELAY_IMPORT, _DELAY_MANGA],
            "src/manga_editor_core.py": [_DELAY_IMPORT, _DELAY_MANGA],
            "src/async_api_processor.py": [_DELAY_ASYNC],
            "src/QA_Scanner_GUI.py": _DELAY_QA_GUI,
            "src/scan_html_folder.py": _DELAY_SCANNER,
        },
        "values": {},
    },
    {
        "date": "2026-10-22",
        "what": "Clean and Translate load the selected local inpainter when no preload filled its pool (model "
                "switched after the startup preload) instead of polling 30 s and failing (Clean)",
        "source": {
            "src/ImageRenderer.py": [_POOL_LOAD_HELPER, _POOL_LOAD_CLEAN, _POOL_LOAD_SYNC],
        },
        "values": {},
    },
    {
        "date": "2026-10-22",
        "what": "A Files batch Start leaves the manual editor's boxes: the automatic OCR save after detection / "
                "translation also writes the page's editor state (detection boxes, OCR and translations, as an "
                "OCR import does), so the editor draws the rectangles when the page is opened",
        "source": {
            "src/manga_integration.py": [_BATCH_EDITOR_STATE_NO_DOC, _BATCH_EDITOR_STATE_DOC],
        },
        "values": {},
    },
    {
        "date": "2026-10-22",
        "what": "Closing the window no longer hangs in the image state manager's worker shutdown: its "
                "queues are closed without waiting for feeder threads the gone worker never drains",
        "source": {
            "src/manga_integration.py": [_STATE_QUEUE_CLOSE],
        },
        "values": {},
    },
    {
        "date": "2026-10-22",
        "what": "New experimental glossary name checks, both off by default: 'Glossary Name Check After "
                "Translation' (Other Settings; Auto-fix or Review first) and the QA scanner's 'Check glossary "
                "names'; they flag translated names spelled differently from the glossary",
        "source": {
            "src/settings_persistence.py": [_NAME_CHECK_SETTINGS_ROW],
            "src/owner_state.py": [_NAME_CHECK_BOOL_ROW, _NAME_CHECK_STR_ROW, _NAME_CHECK_MIN_CHARS_INIT],
            "src/translator_gui.py": [_NAME_CHECK_SETTINGS_ROW_GUI, _NAME_CHECK_BOOL_ROW, _NAME_CHECK_STR_ROW,
                                      _NAME_CHECK_ENV_DICT, _NAME_CHECK_QA_LABEL, _NAME_CHECK_QA_ENV_PAIR],
            "src/run_env.py": [_NAME_CHECK_ENV_DICT, _NAME_CHECK_QA_LABEL, _NAME_CHECK_QA_ENV_PAIR],
            "src/qa_scan_runtime.py": _NAME_CHECK_QA_RUNTIME,
            "src/QA_Scanner_GUI.py": _NAME_CHECK_QA_GUI,
            "src/scan_html_folder.py": _NAME_CHECK_SCANNER,
            "src/TransateKRtoEN.py": [_NAME_CHECK_WORKER_QA],
        },
        # new keys: nothing recorded before they existed changes value
        "values": {},
    },
    {
        "date": "2026-10-23",
        "what": "Manga Translate: with no preload filling the inpainter pool (Glossarion Mobile, or a model "
                "switched after startup) the concurrent inpainting loads one right away instead of waiting 60 s "
                "('Waiting for preloaded inpainter...'); Clean shares the check; 'Locating <model> model' "
                "replaces the 'Downloading <model> model' line logged when the model was already cached",
        "source": {
            "src/ImageRenderer.py": _INPAINT_NO_POOL_SYNC,
        },
        "values": {},
    },
)


def apply_to_source(relpath: str, text: str) -> str:
    """Legacy source ``relpath`` (as ``git show`` returns it) with the approved changes applied."""
    relpath = str(relpath).replace("\\", "/")
    for change in APPROVED:
        for old, new in change["source"].get(relpath, ()):
            text = text.replace(old, new)
    return text


def _changed(value, old, new):
    if value == old and type(value) is type(old):
        return new
    if isinstance(value, str) and value == str(old):  # values recorded as text
        return str(new)
    return value


def apply_to_golden(data):
    """A loaded golden (nested dicts / lists) with the approved recorded values changed and the
    approved new keys inserted."""
    values = {}
    inserts = []
    for change in APPROVED:
        values.update(change["values"])
        inserts.extend(change.get("inserts", ()))

    def _inserted(parent, node):
        for rule in inserts:
            if parent in rule["parents"] and rule["after"] in node and rule["key"] not in node:
                out = {}
                for key, value in node.items():
                    out[key] = value
                    if key == rule["after"]:
                        out[rule["key"]] = rule["value"]
                node = out
        return node

    def walk(node, parent=None):
        if isinstance(node, dict):
            node = _inserted(parent, node)
            return {key: (_changed(walk(value, key), *values[key]) if key in values else walk(value, key))
                    for key, value in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        return node

    return walk(data)
