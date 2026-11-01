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
# 2026-10-23 "Payload cleanup limit (MB)" (payload_cleanup_limit_mb, 400; 0 = never clean up)
_PAYLOAD_CLEANUP_ROWS = ("('payload_cleanup_limit_mb', ['payload_cleanup_limit_mb_var'], 400, "
                         "lambda v: max(0, safe_int(v, 400))),",)
_PAYLOAD_CLEANUP_SETTINGS_ROW = _after(
    "            ('glossary_name_check_autofix_level', ['glossary_name_check_autofix_level_var'], 1, lambda v: "
    "max(1, min(3, safe_int(v, 1)))),", *("            " + row for row in _PAYLOAD_CLEANUP_ROWS))
_PAYLOAD_CLEANUP_SETTINGS_ROW_GUI = _after(
    "                ('glossary_name_check_autofix_level', ['glossary_name_check_autofix_level_var'], 1, lambda v: "
    "max(1, min(3, safe_int(v, 1)))),", *("                " + row for row in _PAYLOAD_CLEANUP_ROWS))
_PAYLOAD_CLEANUP_INIT = _after(
    "            self.glossary_name_check_autofix_level_var = 1",
    "        try:",
    "            self.payload_cleanup_limit_mb_var = max(0, int(self.config.get('payload_cleanup_limit_mb', 400)))",
    "        except (TypeError, ValueError):",
    "            self.payload_cleanup_limit_mb_var = 400")
# 2026-10-23 Payloads > "Auto-update": the generated schema rows (oracle pinned at HEAD reads the tables from it)
_PAYLOAD_SUMMARY_AUTO_SCHEMA = [('        "only reported, never fixed automatically, unless they differ just in spacing or hyphens."),\n    "payload_cleanup_limit_mb": (\n', '        "only reported, never fixed automatically, unless they differ just in spacing or hyphens."),\n    "payload_summary_auto_update": (\n        "Rebuild the readable payload summary (Payload Summary/logistics.html) after a translation or "\n        "glossary run is stopped, once the graceful stop has finished."),\n    "payload_cleanup_limit_mb": (\n')]
_PAYLOAD_SUMMARY_AUTO_SCHEMA_DATA = [(" 'payload_cleanup_limit_mb',\n 'glossary_name_check_use_ner',\n", " 'payload_cleanup_limit_mb',\n 'payload_summary_auto_update',\n 'glossary_name_check_use_ner',\n"), ("  ('safe_int', 400, 0, None, '')),\n ('glossary_name_check_use_ner', ('glossary_name_check_use_ner_var',), ('value', True), ('bool',)),\n", "  ('safe_int', 400, 0, None, '')),\n ('payload_summary_auto_update', ('payload_summary_auto_update_var',), ('value', False), ('bool',)),\n ('glossary_name_check_use_ner', ('glossary_name_check_use_ner_var',), ('value', True), ('bool',)),\n"), (" ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', ('value', True)),\n ('contextual_var', 'contextual', ('value', False)),\n", " ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', ('value', True)),\n ('payload_summary_auto_update_var', 'payload_summary_auto_update', ('value', False)),\n ('contextual_var', 'contextual', ('value', False)),\n"), ("     'var_names': ('payload_cleanup_limit_mb_var',)},\n    'pdf_async_page_threshold': {'converter': ('str',),\n", "     'var_names': ('payload_cleanup_limit_mb_var',)},\n    'payload_summary_auto_update': {'converter': ('bool',),\n     'default': False,\n     'default_source': 'init',\n     'dialog_default': False,\n     'init_default': False,\n     'label': 'Auto-update',\n     'origins': ('settings_map', 'init', 'dialog'),\n     'save_default': False,\n     'tooltip': 'Rebuild the payload summary in the background whenever a translation or glossary '\n                'run is stopped, after the graceful stop has finished (the log shows where it was '\n                'written).',\n     'type': 'bool',\n     'ui_group': 'Payloads',\n     'ui_sites': ('other.dialog',),\n     'var_names': ('payload_summary_auto_update_var',)},\n    'pdf_async_page_threshold': {'converter': ('str',),\n")]
# 2026-10-23 Auto-update also after a finished translation / glossary extraction (tooltip, the two calls);
# Payloads > "Decode images" (payload_summary_decode_images, off)
_PAYLOAD_SUMMARY_FINISH_SCHEMA = [('    "payload_summary_auto_update": (\n        "Rebuild the readable payload summary (Payload Summary/logistics.html) after a translation or "\n        "glossary run is stopped, once the graceful stop has finished."),\n    "payload_cleanup_limit_mb": (\n', '    "payload_summary_auto_update": (\n        "Rebuild the readable payload summary (Payload Summary/logistics.html) when a translation or "\n        "glossary extraction finishes, and when one is stopped, once the graceful stop has finished."),\n    "payload_summary_decode_images": (\n        "The payload summary\'s detail pages show the images each request sent (decoded into "\n        "Payload Summary/details/images, about three quarters of their base64 size on disk)."),\n    "payload_cleanup_limit_mb": (\n')]
_PAYLOAD_SUMMARY_FINISH_SCHEMA_DATA = [(" 'payload_summary_auto_update',\n 'glossary_name_check_use_ner',\n", " 'payload_summary_auto_update',\n 'payload_summary_decode_images',\n 'glossary_name_check_use_ner',\n"), (" ('payload_summary_auto_update', ('payload_summary_auto_update_var',), ('value', False), ('bool',)),\n ('glossary_name_check_use_ner', ('glossary_name_check_use_ner_var',), ('value', True), ('bool',)),\n", " ('payload_summary_auto_update', ('payload_summary_auto_update_var',), ('value', False), ('bool',)),\n ('payload_summary_decode_images',\n  ('payload_summary_decode_images_var',),\n  ('value', False),\n  ('bool',)),\n ('glossary_name_check_use_ner', ('glossary_name_check_use_ner_var',), ('value', True), ('bool',)),\n"), (" ('payload_summary_auto_update_var', 'payload_summary_auto_update', ('value', False)),\n ('contextual_var', 'contextual', ('value', False)),\n", " ('payload_summary_auto_update_var', 'payload_summary_auto_update', ('value', False)),\n ('payload_summary_decode_images_var', 'payload_summary_decode_images', ('value', False)),\n ('contextual_var', 'contextual', ('value', False)),\n"), ("     'save_default': False,\n     'tooltip': 'Rebuild the payload summary in the background whenever a translation or glossary '\n                'run is stopped, after the graceful stop has finished (the log shows where it was '\n                'written).',\n     'type': 'bool',\n", '     \'save_default\': False,\n     \'tooltip\': \'Rebuild the payload summary in the background when a translation or glossary \'\n                "extraction finishes (after \'Translation completed\' / the glossary extraction "\n                \'summary) and when one is stopped, once the graceful stop has finished (the log \'\n                \'shows where it was written).\',\n     \'type\': \'bool\',\n'), ("     'var_names': ('payload_summary_auto_update_var',)},\n    'pdf_async_page_threshold': {'converter': ('str',),\n", '     \'var_names\': (\'payload_summary_auto_update_var\',)},\n    \'payload_summary_decode_images\': {\'converter\': (\'bool\',),\n     \'default\': False,\n     \'default_source\': \'init\',\n     \'dialog_default\': False,\n     \'init_default\': False,\n     \'label\': \'Decode images\',\n     \'origins\': (\'settings_map\', \'init\', \'dialog\'),\n     \'save_default\': False,\n     \'tooltip\': \'Show the images each request sent (manga pages, inpainting, image translation) on \'\n                "its detail page. They are decoded into \'Payload Summary/details/images\', which "\n                \'takes about three quarters of their size in the payloads on disk and is replaced \'\n                \'on every rebuild.\',\n     \'type\': \'bool\',\n     \'ui_group\': \'Payloads\',\n     \'ui_sites\': (\'other.dialog\',),\n     \'var_names\': (\'payload_summary_decode_images_var\',)},\n    \'pdf_async_page_threshold\': {\'converter\': (\'str\',),\n')]
_PAYLOAD_SUMMARY_FINISH_TEXT_JOBS = [('                        self.append_log("✅ Translation completed successfully!")\n                    return True\n', '                        self.append_log("✅ Translation completed successfully!")\n                    # Payloads > Auto-update (desktop setting): this run\'s payloads into the readable summary\n                    import payload_summary\n                    payload_summary.auto_update(self)\n                    return True\n'), ('                        self.append_log("✅ Translation completed successfully!")\n                    # Payloads > Auto-update: this run\'s payloads into the readable summary (desktop only)\n                    getattr(self, \'_auto_update_payload_summary\', lambda: None)()\n                    return True\n', '                        self.append_log("✅ Translation completed successfully!")\n                    # Payloads > Auto-update (desktop setting): this run\'s payloads into the readable summary\n                    import payload_summary\n                    payload_summary.auto_update(self)\n                    return True\n')]
_PAYLOAD_SUMMARY_FINISH_PIPELINE = [('            self.append_log(f"   📁 Total processed: {total_groups}")\n            self.append_log(f"{\'=\'*60}")\n            \n        except Exception as e:\n', '            self.append_log(f"   📁 Total processed: {total_groups}")\n            self.append_log(f"{\'=\'*60}")\n            # Payloads > Auto-update (desktop setting): this run\'s payloads into the readable summary\n            import payload_summary\n            payload_summary.auto_update(self)\n            \n        except Exception as e:\n'), ('            self.append_log(f"{\'=\'*60}")\n            # Payloads > Auto-update: this run\'s payloads into the readable summary (desktop only)\n            getattr(self, \'_auto_update_payload_summary\', lambda: None)()\n\n        except Exception as e:\n', '            self.append_log(f"{\'=\'*60}")\n            # Payloads > Auto-update (desktop setting): this run\'s payloads into the readable summary\n            import payload_summary\n            payload_summary.auto_update(self)\n            \n        except Exception as e:\n')]
# the same, in the pinned older sources (settings / bool-var rows, the two calls in translator_gui)
_PAYLOAD_SUMMARY_LEGACY_OWNER_STATE = [("            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('contextual_var', 'contextual', False),\n", "            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n            ('payload_summary_decode_images_var', 'payload_summary_decode_images', False),\n            ('contextual_var', 'contextual', False),\n")]
_PAYLOAD_SUMMARY_LEGACY_SETTINGS_PERSISTENCE = [("            ('payload_cleanup_limit_mb', ['payload_cleanup_limit_mb_var'], 400, lambda v: max(0, safe_int(v, 400))),\n            ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n", "            ('payload_cleanup_limit_mb', ['payload_cleanup_limit_mb_var'], 400, lambda v: max(0, safe_int(v, 400))),\n            ('payload_summary_auto_update', ['payload_summary_auto_update_var'], False, bool),\n            ('payload_summary_decode_images', ['payload_summary_decode_images_var'], False, bool),\n            ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n")]
_PAYLOAD_SUMMARY_LEGACY_TRANSLATOR_GUI = [("            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('contextual_var', 'contextual', False),\n", "            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n            ('payload_summary_decode_images_var', 'payload_summary_decode_images', False),\n            ('contextual_var', 'contextual', False),\n"), ("                ('payload_cleanup_limit_mb', ['payload_cleanup_limit_mb_var'], 400, lambda v: max(0, safe_int(v, 400))),\n                ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n", "                ('payload_cleanup_limit_mb', ['payload_cleanup_limit_mb_var'], 400, lambda v: max(0, safe_int(v, 400))),\n                ('payload_summary_auto_update', ['payload_summary_auto_update_var'], False, bool),\n                ('payload_summary_decode_images', ['payload_summary_decode_images_var'], False, bool),\n                ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n"), ('                    else:\n                        self.append_log("✅ Translation completed successfully!")\n                    return True\n', '                    else:\n                        self.append_log("✅ Translation completed successfully!")\n                    # Payloads > Auto-update (desktop setting): this run\'s payloads into the readable summary\n                    import payload_summary\n                    payload_summary.auto_update(self)\n                    return True\n'), ('            self.append_log(f"   📁 Total processed: {total_groups}")\n            self.append_log(f"{\'=\'*60}")\n            \n        except Exception as e:\n', '            self.append_log(f"   📁 Total processed: {total_groups}")\n            self.append_log(f"{\'=\'*60}")\n            # Payloads > Auto-update (desktop setting): this run\'s payloads into the readable summary\n            import payload_summary\n            payload_summary.auto_update(self)\n            \n        except Exception as e:\n')]
# 2026-10-24 CI: the glossary name check / payload cleanup settings in the pinned translator_gui
_OLD_SETTINGS_INIT_GUI = [('            self.emergency_glossary_compliance_min_chars_var = 3\n        \n        # NEW: Initialize extraction mode variable\n', "            self.emergency_glossary_compliance_min_chars_var = 3\n        try:\n            self.glossary_name_check_min_chars_var = max(0, min(10, int(self.config.get('glossary_name_check_min_chars', 2))))\n        except (TypeError, ValueError):\n            self.glossary_name_check_min_chars_var = 2\n        try:\n            self.glossary_name_check_autofix_level_var = max(1, min(3, int(self.config.get('glossary_name_check_autofix_level', 1))))\n        except (TypeError, ValueError):\n            self.glossary_name_check_autofix_level_var = 1\n        try:\n            self.payload_cleanup_limit_mb_var = max(0, int(self.config.get('payload_cleanup_limit_mb', 400)))\n        except (TypeError, ValueError):\n            self.payload_cleanup_limit_mb_var = 400\n        \n        # NEW: Initialize extraction mode variable\n")]
_PAYLOAD_CLEANUP_SWEEP_GUI = [('        return (0, 0, 0, 0)\n\n\ndef _sweep_large_caches(max_bytes: int = 400 * 1024 * 1024, phase: str = "startup") -> None:\n', '        return (0, 0, 0, 0)\n\n\ndef payload_cleanup_limit_bytes(default=400 * 1024 * 1024, config_file=None):\n    """The debug-cache cap from Other Settings › "Payload cleanup limit (MB)" (``payload_cleanup_limit_mb``\n    in config.json; 0 = never clean up -> None); ``default`` when the config has no such key. Read\n    from the saved config because the startup sweep runs before the GUI has loaded it."""\n    try:\n        if config_file is None:\n            from app_paths import CONFIG_FILE as config_file\n        with open(config_file, "r", encoding="utf-8") as f:\n            value = json.load(f).get("payload_cleanup_limit_mb")\n        if value is None:\n            return default\n        megabytes = int(value)\n        return None if megabytes <= 0 else megabytes * 1024 * 1024\n    except Exception:\n        return default\n\n\ndef _sweep_large_caches(max_bytes: int = 400 * 1024 * 1024, phase: str = "startup") -> None:\n'), ('    moment the first dump is saved.\n\n    ``phase`` is just a label', '    moment the first dump is saved.\n\n    The cap is the saved "Payload cleanup limit (MB)" setting when there is one\n    (``payload_cleanup_limit_bytes``; 0 = never clean up), else ``max_bytes``.\n\n    ``phase`` is just a label'), ('    """\n    try:\n        # Candidate roots: exe dir (frozen), script dir (dev), and CWD.\n', '    """\n    max_bytes = payload_cleanup_limit_bytes(max_bytes)\n    if max_bytes is None:\n        return  # "0 = never clean up"\n    try:\n        # Candidate roots: exe dir (frozen), script dir (dev), and CWD.\n')]
# 2026-10-24 Arena reCAPTCHA rejections: no full traceback (like config / entitlement errors)
_ARENA_VERIFICATION_NO_TRACEBACK = [("is_config_error = getattr(e, 'error_type', None) in ('config_error', 'entitlement_error')", "is_config_error = getattr(e, 'error_type', None) in ('config_error', 'entitlement_error', 'autharena_verification_error')")]
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
  '        logs = []\n'
  '\n'
  '        def work(state, cancelled):\n'
  '            def on_download(report):\n'
  '                speed = f" · {report.bytes_per_sec / mb:.1f} MB/s" if report.bytes_per_sec else ""\n'
  '                text = ("Verifying the named-entity model (NER)..." if report.phase == "verify" else\n'
  '                        f"Downloading the named-entity model (NER): {report.downloaded / mb:.1f} / "\n'
  '                        f"{report.total / mb:.1f} MB{speed}")\n'
  '                state.update(text=text, value=report.percent, maximum=100)\n'
  '\n'
  '            where = {"text": ""}\n'
  '\n'
  '            def on_chapter(done, total):\n'
  '                state.update(text=f"{where[\'text\']}Checking chapter {done + 1} of {total}...", '
  'value=done,\n'
  '                             maximum=total)\n'
  '\n'
  '            ner = name_check.resolve_ner(use_ner, logs.append, progress=on_download, cancel=cancelled, retry=True)\n'
  '            workers = name_check.chapter_workers(ner, chosen=name_check.workers_from_config(\n'
  '                {"glossary_name_check_workers": getattr(self, \'glossary_name_check_workers_var\',\n'
  "                                                        self.config.get('glossary_name_check_workers', 0))}))\n"
  '            reviews = []\n'
  '            for index, folder in enumerate(folders):\n'
  '                if cancelled():\n'
  '                    break\n'
  '                if len(folders) > 1:\n'
  '                    where["text"] = f"{os.path.basename(folder)} ({index + 1} of {len(folders)})\\n"\n'
  '                state.update(text=f"{where[\'text\']}Checking glossary names...", value=0, maximum=0)\n'
  '                reviews.append(name_check.review_folder(folder, scope, stop=cancelled, ner=ner,\n'
  '                                                        progress=on_chapter, workers=workers,\n'
  '                                                        settings=layer_settings))\n'
  '            return reviews\n'
  '\n'
  '        def flush_logs():\n'
  '            while logs:\n'
  '                line = logs.pop(0)\n'
  '                if log:\n'
  '                    log(line)\n'
  '\n'
  '        from qt_widget_helpers import run_with_progress\n'
  '        return run_with_progress(parent, title, work, text="Checking glossary names...", '
  'on_tick=flush_logs,\n'
  '                                 thread_name="GlossaryNameReview")\n'
  '\n'
  '    def _glossary_name_check_scope(self):\n'
  '        """Emergency Glossary Compliance\'s entry types and the name check\'s minimum length (live '
  'values)."""\n'
  '        import glossary_name_check as name_check\n'
  '        config = dict(self.config or {})\n'
  "        for key in ('emergency_glossary_compliance_mode', 'emergency_glossary_compliance_custom_types',\n"
  "                    'glossary_name_check_min_chars', 'glossary_name_check_autofix_level',\n"
  "                    'glossary_name_check_match_numbers', 'glossary_name_check_plural_exclusion',\n"
  "                    'glossary_name_check_tense_exclusion', 'glossary_name_check_inflection_skip_types'):\n"
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
  '            # the label says whether NER does the check (owner 2026-10-27: ✓ when its model is installed); it\n'
  '            # follows "Use Named Entity Recognition model (NER)" below as that changes (owner 2026-10-28)\n'
  '            import glossary_name_check as name_check\n'
  '            ner_labels = {\n'
  '                "installed": "Check glossary names with NER (Experimental)",\n'
  '                "missing": "Check glossary names (Experimental) — NER not installed (downloads on first use)",\n'
  '                "unsupported": "Check glossary names (Experimental) — NER not supported in this build",\n'
  '                "off": "Check glossary names (Experimental) — NER off",\n'
  '            }\n'
  '            check_glossary_names_checkbox = self._create_styled_checkbox(ner_labels["installed"])\n'
  "            check_glossary_names_checkbox.setChecked(qa_settings.get('check_glossary_names', False))\n"
  '            check_glossary_names_checkbox.setToolTip(\n'
  '                "Flags chapters where a glossary name from the source chapter was translated with a\\n"\n'
  '                "different spelling (e.g. \'Walter Argent\' for glossary \'Valter Argent\').\\n"\n'
  '                "Names are found with a named-entity recognition model (NER, 104 MB, downloaded on\\n"\n'
  '                "first use); \\"Use Named Entity Recognition model (NER)\\" below turns it off for a\\n"\n'
  '                "capitalization heuristic (builds without ONNX Runtime always use the heuristic).\\n"\n'
  '                "The settings below are the same as the name check\'s in Other Settings (a change in\\n"\n'
  '                "either shows in both). Flagged chapters count as QA failures like other issues.")\n'
  '            check_glossary_names_row = QWidget()\n'
  '            check_glossary_names_layout = QHBoxLayout(check_glossary_names_row)\n'
  '            check_glossary_names_layout.setContentsMargins(0, 0, 0, 0)\n'
  '            check_glossary_names_layout.addWidget(check_glossary_names_checkbox)\n'
  '            ner_installed_icon = QLabel("✓")\n'
  '            ner_installed_icon.setStyleSheet("color: #4caf50; font-weight: bold; font-size: 12pt;")\n'
  '            ner_installed_icon.setToolTip("The named-entity model (NER) is installed")\n'
  '            check_glossary_names_layout.addWidget(ner_installed_icon)\n'
  '            check_glossary_names_layout.addStretch()\n'
  '            detection_layout.addWidget(check_glossary_names_row)\n'
  '\n'
  '            def _refresh_ner_label(_checked=None):\n'
  "                state = name_check.ner_status(bool(getattr(self, 'glossary_name_check_use_ner_var',\n"
  "                                                           self.config.get('glossary_name_check_use_ner', True))))\n"
  '                check_glossary_names_checkbox.setText(ner_labels[state])\n'
  '                ner_installed_icon.setVisible(state == "installed")\n'
  '            _refresh_ner_label()\n'
  '\n'
  "            # Other Settings' name check settings, one setting each (a change in either dialog shows in both):\n"
  '            # entry types, Auto-fix strength, Chapters checked at once, Names keep their numbers, plural / tense\n'
  '            # exclusion (and the types they skip) and Use NER\n'
  '            from other_settings import (glossary_entry_types_row, glossary_name_autofix_row,\n'
  '                                        glossary_name_inflection_types_row, glossary_name_ner_row,\n'
  '                                        glossary_name_numbers_row, glossary_name_plural_row, glossary_name_tense_row,\n'
  '                                        glossary_name_workers_row)\n'
  '            for glossary_names_setting_row in (glossary_entry_types_row(self, lambda: dialog),\n'
  '                                               glossary_name_autofix_row(self), glossary_name_workers_row(self),\n'
  '                                               glossary_name_numbers_row(self), glossary_name_plural_row(self),\n'
  '                                               glossary_name_tense_row(self),\n'
  '                                               glossary_name_inflection_types_row(self, lambda: dialog),\n'
  '                                               glossary_name_ner_row(self, on_change=_refresh_ner_label)):\n'
  '                detection_layout.addWidget(glossary_names_setting_row)\n'
  '                glossary_names_setting_row.setVisible(check_glossary_names_checkbox.isChecked())\n'
  '                check_glossary_names_checkbox.toggled.connect(glossary_names_setting_row.setVisible)\n'
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
 ('            # AI Hunter Performance Section\n'
  '            ai_hunter_group = QGroupBox("AI Hunter Performance Settings")\n'
  "            ai_hunter_group.setFont(QFont('Arial', 12, QFont.Bold))\n",
  '            # AI Hunter Performance Section\n'
  '            ai_hunter_group = QGroupBox("Scanner Performance Settings")\n'
  "            ai_hunter_group.setFont(QFont('Arial', 12, QFont.Bold))\n"),
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
  '            from glossary_name_check import (layer_settings_from_config, settings_from_config, use_ner_from_config,\n'
  '                                             workers_from_config)\n'
  '            settings["_glossary_name_scope"] = settings_from_config(live_config)\n'
  '            settings["_glossary_name_use_ner"] = use_ner_from_config(live_config)\n'
  '            settings["_glossary_name_workers"] = workers_from_config(live_config)  # 0 = Auto\n'
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
  '    workers = name_check.chapter_workers(ner, chosen=qa_settings.get("_glossary_name_workers") or 0)\n'
  '    log(f"   🔎 Names found with "\n'
  '        + (f"the named-entity model (NER), {workers} chapter(s) at once" if ner is not None\n'
  '           else "the capitalization heuristic"))\n'
  '    # progress: a line every tenth of the chapters (at most every 15 s apart) and the last one\n'
  '    shown = {"done": 0, "at": time.monotonic()}\n'
  '    shown_lock = threading.Lock()\n'
  '\n'
  '    def on_progress(index, total):\n'
  '        done = index + 1\n'
  '        with shown_lock:\n'
  '            now = time.monotonic()\n'
  '            if done <= shown["done"] or not (done == total or done % max(1, total // 10) == 0\n'
  '                                             or now - shown["at"] >= 15):\n'
  '                return\n'
  '            shown["done"], shown["at"] = done, now\n'
  '        log(f"   📖 Glossary name check: {done}/{total} chapters ({done * 100 // max(1, total)}%)")\n'
  '\n'
  '    chapters = name_check.scan_folder(folder_path, entries, progress_path=progress_path,\n'
  '                                      selected_files=list(by_file), stop=should_stop, ner=ner, progress=on_progress,\n'
  '                                      workers=workers)\n'
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

# 2026-10-27 "Names keep their numbers" (glossary_name_check_match_numbers / GLOSSARY_NAME_CHECK_MATCH_NUMBERS,
# on): the settings / bool rows and the env line in the pinned sources, and the schema hunks
_MATCH_NUMBERS_LEGACY = {'src/translator_gui.py': [("                ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "                ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n                ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n            ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n", "            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n"), ('            \'GLOSSARY_NAME_CHECK_USE_NER\': "1" if getattr(self, \'glossary_name_check_use_ner_var\', True) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n', '            \'GLOSSARY_NAME_CHECK_USE_NER\': "1" if getattr(self, \'glossary_name_check_use_ner_var\', True) else "0",\n            \'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\': "1" if getattr(self, \'glossary_name_check_match_numbers_var\', True) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n')], 'src/owner_state.py': [("            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n", "            ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', True),\n            ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n")], 'src/settings_persistence.py': [("            ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_use_ner', ['glossary_name_check_use_ner_var'], True, bool),\n            ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n")], 'src/run_env.py': [('            \'GLOSSARY_NAME_CHECK_USE_NER\': "1" if getattr(self, \'glossary_name_check_use_ner_var\', True) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n', '            \'GLOSSARY_NAME_CHECK_USE_NER\': "1" if getattr(self, \'glossary_name_check_use_ner_var\', True) else "0",\n            \'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\': "1" if getattr(self, \'glossary_name_check_match_numbers_var\', True) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n')]}
_MATCH_NUMBERS_SCHEMA = {'src/settings_schema_data.py': [(" 'glossary_name_check_use_ner',\n 'retry_duplicate_bodies',\n", " 'glossary_name_check_use_ner',\n 'glossary_name_check_match_numbers',\n 'retry_duplicate_bodies',\n"), (" ('glossary_name_check_use_ner', ('glossary_name_check_use_ner_var',), ('value', True), ('bool',)),\n ('retry_duplicate_bodies', ('retry_duplicate_var',), ('value', False), ('bool',)),\n", " ('glossary_name_check_use_ner', ('glossary_name_check_use_ner_var',), ('value', True), ('bool',)),\n ('glossary_name_check_match_numbers',\n  ('glossary_name_check_match_numbers_var',),\n  ('value', True),\n  ('bool',)),\n ('retry_duplicate_bodies', ('retry_duplicate_var',), ('value', False), ('bool',)),\n"), (" ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', ('value', True)),\n ('payload_summary_auto_update_var', 'payload_summary_auto_update', ('value', False)),\n", " ('glossary_name_check_use_ner_var', 'glossary_name_check_use_ner', ('value', True)),\n ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', ('value', True)),\n ('payload_summary_auto_update_var', 'payload_summary_auto_update', ('value', False)),\n"), ("     'var_names': ('glossary_name_check_autofix_level_var',)},\n    'glossary_name_check_min_chars': {'converter': ('safe_int', 2, 0, 10, 'max_min'),\n", '     \'var_names\': (\'glossary_name_check_autofix_level_var\',)},\n    \'glossary_name_check_match_numbers\': {\'converter\': (\'bool\',),\n     \'default\': True,\n     \'default_source\': \'init\',\n     \'dialog_default\': True,\n     \'env\': ((\'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\', \'translation\', True),),\n     \'init_default\': True,\n     \'label\': \'Names keep their numbers\',\n     \'origins\': (\'settings_map\', \'init\', \'dialog\', \'env\'),\n     \'save_default\': True,\n     \'tooltip\': \'A spelling with another number, or none, is never taken for a glossary name\\n\'\n                "(\'Demon Kings\' is not \'72 Demon Kings\'), and a name whose number is glued to the\\n"\n                "word before it (\'the72 Demon Kings\', a missing space) counts as found.\\n"\n                \'Off: numbers are ignored when names are compared.\',\n     \'type\': \'bool\',\n     \'ui_group\': \'Processing Options\',\n     \'ui_sites\': (\'other.processing\',),\n     \'var_names\': (\'glossary_name_check_match_numbers_var\',)},\n    \'glossary_name_check_min_chars\': {\'converter\': (\'safe_int\', 2, 0, 10, \'max_min\'),\n'), ("             ('QA_CHECK_GLOSSARY_NAMES', 'qa', False)),\n     'label': 'Check glossary names with NER (Experimental)',\n     'nested_default': False,\n", "             ('QA_CHECK_GLOSSARY_NAMES', 'qa', False)),\n     'nested_default': False,\n")], 'src/settings_schema.py': [('    "glossary_name_check_mode": "Glossary name check mode",\n    "output_language": "Target language",\n', '    "glossary_name_check_mode": "Glossary name check mode",\n    # the QA scanner\'s toggle text says whether NER is installed (picked at runtime, 2026-10-27)\n    "check_glossary_names": "Check glossary names with NER (Experimental)",\n    "output_language": "Target language",\n')]}


# 2026-10-27 "Chapters checked at once" (glossary_name_check_workers, 0 = Auto): the settings row in the
# pinned sources (after the match-numbers row)
_NAME_CHECK_WORKERS_LEGACY = {'src/translator_gui.py': [("                ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "                ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n                ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n            ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n")], 'src/settings_persistence.py': [("            ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_match_numbers', ['glossary_name_check_match_numbers_var'], True, bool),\n            ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n")]}


# 2026-10-28 "NER plural exclusion" (glossary_name_check_plural_exclusion, on) and "NER tense exclusion"
# (glossary_name_check_tense_exclusion, off; English output only): rows and env lines in the pinned sources
_NAME_CHECK_INFLECTION_LEGACY = {'src/translator_gui.py': [("                ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "                ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n                ('glossary_name_check_plural_exclusion', ['glossary_name_check_plural_exclusion_var'], True, bool),\n                ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n            ('glossary_name_check_plural_exclusion', ['glossary_name_check_plural_exclusion_var'], True, bool),\n            ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n", "            ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', True),\n            ('glossary_name_check_plural_exclusion_var', 'glossary_name_check_plural_exclusion', True),\n            ('glossary_name_check_tense_exclusion_var', 'glossary_name_check_tense_exclusion', False),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n"), ('            \'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\': "1" if getattr(self, \'glossary_name_check_match_numbers_var\', True) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n', '            \'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\': "1" if getattr(self, \'glossary_name_check_match_numbers_var\', True) else "0",\n            \'GLOSSARY_NAME_CHECK_PLURAL_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_plural_exclusion_var\', True) else "0",\n            \'GLOSSARY_NAME_CHECK_TENSE_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_tense_exclusion_var\', False) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n'), ("                ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "                ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n                ('glossary_name_check_inflection_skip_types', ['glossary_name_check_inflection_skip_types_var'], ['character', 'nicknames'], list),\n                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n            ('glossary_name_check_inflection_skip_types', ['glossary_name_check_inflection_skip_types_var'], ['character', 'nicknames'], list),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ('            \'GLOSSARY_NAME_CHECK_TENSE_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_tense_exclusion_var\', False) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n', '            \'GLOSSARY_NAME_CHECK_TENSE_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_tense_exclusion_var\', False) else "0",\n            \'GLOSSARY_NAME_CHECK_INFLECTION_SKIP_TYPES\': json.dumps(getattr(self, \'glossary_name_check_inflection_skip_types_var\', self.config.get(\'glossary_name_check_inflection_skip_types\', [\'character\', \'nicknames\']))),\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n')], 'src/owner_state.py': [("            ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', True),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n", "            ('glossary_name_check_match_numbers_var', 'glossary_name_check_match_numbers', True),\n            ('glossary_name_check_plural_exclusion_var', 'glossary_name_check_plural_exclusion', True),\n            ('glossary_name_check_tense_exclusion_var', 'glossary_name_check_tense_exclusion', False),\n            ('payload_summary_auto_update_var', 'payload_summary_auto_update', False),\n")], 'src/settings_persistence.py': [("            ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_workers', ['glossary_name_check_workers_var'], 0, lambda v: max(0, min(32, safe_int(v, 0)))),\n            ('glossary_name_check_plural_exclusion', ['glossary_name_check_plural_exclusion_var'], True, bool),\n            ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n"), ("            ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n", "            ('glossary_name_check_tense_exclusion', ['glossary_name_check_tense_exclusion_var'], False, bool),\n            ('glossary_name_check_inflection_skip_types', ['glossary_name_check_inflection_skip_types_var'], ['character', 'nicknames'], list),\n            ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),\n")], 'src/run_env.py': [('            \'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\': "1" if getattr(self, \'glossary_name_check_match_numbers_var\', True) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n', '            \'GLOSSARY_NAME_CHECK_MATCH_NUMBERS\': "1" if getattr(self, \'glossary_name_check_match_numbers_var\', True) else "0",\n            \'GLOSSARY_NAME_CHECK_PLURAL_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_plural_exclusion_var\', True) else "0",\n            \'GLOSSARY_NAME_CHECK_TENSE_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_tense_exclusion_var\', False) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n'), ('            \'GLOSSARY_NAME_CHECK_TENSE_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_tense_exclusion_var\', False) else "0",\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n', '            \'GLOSSARY_NAME_CHECK_TENSE_EXCLUSION\': "1" if getattr(self, \'glossary_name_check_tense_exclusion_var\', False) else "0",\n            \'GLOSSARY_NAME_CHECK_INFLECTION_SKIP_TYPES\': json.dumps(getattr(self, \'glossary_name_check_inflection_skip_types_var\', self.config.get(\'glossary_name_check_inflection_skip_types\', [\'character\', \'nicknames\']))),\n            \'COMPRESS_GLOSSARY_PROMPT\': \'1\' if self.config.get(\'compress_glossary_prompt\') else \'0\',\n')]}


_QA_BROKEN_POOL_SCANNER = [('    Uses environment var QA_RESERVE_CORES to decide how many logical cores to leave free (default 1).\n    """\n    try:\n', '    Uses environment var QA_RESERVE_CORES to decide how many logical cores to leave free (default 1).\n    A worker that dies in native code (no memory left, a crashing library) writes its traceback to the\n    app\'s crash.log (the GUI exports GLOSSARION_LOG_DIR; a dead worker breaks the whole pool).\n    """\n    global _WORKER_FAULT_LOG\n    try:\n        log_dir = os.environ.get("GLOSSARION_LOG_DIR")\n        if log_dir and _WORKER_FAULT_LOG is None:\n            import faulthandler\n            _WORKER_FAULT_LOG = open(os.path.join(log_dir, "crash.log"), "a", encoding="utf-8")\n            faulthandler.enable(file=_WORKER_FAULT_LOG, all_threads=True)\n    except Exception:\n        pass\n    try:\n'), ('def _init_worker_process():\n', "_WORKER_FAULT_LOG = None  # crash.log, open for a QA scan worker's lifetime (_init_worker_process)\n\n\ndef _init_worker_process():\n"), ('    # Process files in parallel\n    results = []\n    processed_count = 0\n    \n    with _ExecutorCls(**_exec_kwargs) as executor:\n        # Submit all batches\n        futures = []\n', '    # Process files in parallel\n    results = []\n    processed_count = 0\n    broken_batches = []  # batches a broken process pool lost (a worker died): run again on threads below\n    \n    with _ExecutorCls(**_exec_kwargs) as executor:\n        # Submit all batches\n        futures = []\n        future_args = {}\n'), ('            future = executor.submit(process_html_file_batch, args)\n            futures.append(future)\n        \n        # Collect results as they complete\n        for completed_idx, future in enumerate(concurrent.futures.as_completed(futures)):\n            if should_stop():\n                log("⛔ QA scan interrupted during processing.")\n', '            future = executor.submit(process_html_file_batch, args)\n            futures.append(future)\n            future_args[future] = args\n        \n        # Collect results as they complete\n        for completed_idx, future in enumerate(concurrent.futures.as_completed(futures)):\n            if should_stop():\n                log("⛔ QA scan interrupted during processing.")\n'), ('                results.extend(batch_results)\n                \n            except Exception as e:\n                log(f"   ❌ Error processing batch: {e}")\n                import traceback\n                log(f"   Traceback: {traceback.format_exc()}")\n    \n    # Clear the progress line (like original)\n', '                results.extend(batch_results)\n                \n            except concurrent.futures.BrokenExecutor as e:\n                # a worker process died (a crash in native code or no memory left): every batch still in\n                # the pool fails with it; they run again once the pool is closed, on threads\n                if not broken_batches:\n                    log(f"   ⚠️ A QA scan worker process stopped unexpectedly ({e})")\n                broken_batches.append(future_args[future])\n            except Exception as e:\n                log(f"   ❌ Error processing batch: {e}")\n                import traceback\n                log(f"   Traceback: {traceback.format_exc()}")\n    \n    if broken_batches and not should_stop():\n        log(f"   🧵 Scanning the {len(broken_batches)} batch(es) it left with threads instead "\n            f"(its cause, if it crashed, is in crash.log in the logs folder)")\n        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:\n            for batch_results in thread_executor.map(process_html_file_batch, broken_batches):\n                if should_stop():\n                    log("⛔ QA scan interrupted during processing.")\n                    thread_executor.shutdown(wait=True, cancel_futures=True)\n                    return\n                processed_count += len(batch_results)\n                results.extend(batch_results)\n        log(f"   ✅ {processed_count}/{len(html_files)} files scanned")\n    \n    # Clear the progress line (like original)\n')]
# 2026-10-29 config.json writes are atomic (app_paths._atomic_json_write): the manga settings
# dialog's direct-write fallbacks no longer truncate config.json when a write fails
_ATOMIC_CONFIG_MANGA_DIALOG = [
    ("                    if hasattr(self.main_gui, 'config_file'):\n"
     "                        with open(self.main_gui.config_file, 'w') as f:\n"
     "                            json.dump(self.config, f, indent=2)\n",
     "                    if hasattr(self.main_gui, 'config_file'):\n"
     "                        from app_paths import _atomic_json_write\n"
     "                        _atomic_json_write(self.main_gui.config_file, self.config)\n"),
    ("            elif hasattr(self.main_gui, 'config_file') and isinstance(self.main_gui.config_file, str):\n"
     "                with open(self.main_gui.config_file, 'w', encoding='utf-8') as f:\n"
     "                    json.dump(self.config, f, ensure_ascii=False, indent=2)\n",
     "            elif hasattr(self.main_gui, 'config_file') and isinstance(self.main_gui.config_file, str):\n"
     "                from app_paths import _atomic_json_write\n"
     "                _atomic_json_write(self.main_gui.config_file, self.config)\n"),
    ("                if hasattr(self.main_gui, 'CONFIG_FILE') and isinstance(self.main_gui.CONFIG_FILE, str):\n"
     "                    with open(self.main_gui.CONFIG_FILE, 'w', encoding='utf-8') as f:\n"
     "                        json.dump(self.config, f, ensure_ascii=False, indent=2)\n",
     "                if hasattr(self.main_gui, 'CONFIG_FILE') and isinstance(self.main_gui.CONFIG_FILE, str):\n"
     "                    from app_paths import _atomic_json_write\n"
     "                    _atomic_json_write(self.main_gui.CONFIG_FILE, self.config)\n"),
]
# ... and the QA scanner's custom-mode save
_ATOMIC_CONFIG_QA_SCANNER = [
    ("                    # Write to file\n"
     "                    with open(config_file, 'w', encoding='utf-8') as f:\n"
     "                        json.dump(encrypted_config, f, ensure_ascii=False, indent=2)\n",
     "                    # Write to file (atomically: a failed write never truncates config.json)\n"
     "                    from app_paths import _atomic_json_write\n"
     "                    _atomic_json_write(config_file, encrypted_config)\n"),
]

# 2026-10-28 "Block other scripts in translations (adaptive, all models)" (block_other_scripts /
# BLOCK_OTHER_SCRIPTS, off): the settings / init rows and the env lines in the pinned sources, and the
# schema hunks of the current pin
_BLOCK_OTHER_SCRIPTS_LEGACY = {'src/translator_gui.py': [("        self.bias_common_words_var = self.config.get('bias_common_words', False)\n        self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)\n\n", "        self.bias_common_words_var = self.config.get('bias_common_words', False)\n        self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)\n        self.block_other_scripts_var = self.config.get('block_other_scripts', False)\n\n"), ("                ('bias_repetitive_phrases', ['bias_repetitive_phrases_var'], False, bool),\n                \n                # Glossary anti-duplicate parameters (separate from translation)\n", "                ('bias_repetitive_phrases', ['bias_repetitive_phrases_var'], False, bool),\n                ('block_other_scripts', ['block_other_scripts_var'], False, bool),\n                \n                # Glossary anti-duplicate parameters (separate from translation)\n"), ("            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n", "            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var else '0',\n            'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n"), ("            'BIAS_REPETITIVE_PHRASES': 'Bias against repetitive phrases',\n            \n            # Azure\n", "            'BIAS_REPETITIVE_PHRASES': 'Bias against repetitive phrases',\n            'BLOCK_OTHER_SCRIPTS': 'Block other scripts in translations',\n            \n            # Azure\n"), ("                ('BIAS_REPETITIVE_PHRASES', '1' if getattr(self, 'bias_repetitive_phrases_var', False) else '0'),\n\n                # Azure API version\n", "                ('BIAS_REPETITIVE_PHRASES', '1' if getattr(self, 'bias_repetitive_phrases_var', False) else '0'),\n                ('BLOCK_OTHER_SCRIPTS', '1' if getattr(self, 'block_other_scripts_var', False) else '0'),\n\n                # Azure API version\n")], 'src/owner_state.py': [("        self.bias_common_words_var = self.config.get('bias_common_words', False)\n        self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)\n\n", "        self.bias_common_words_var = self.config.get('bias_common_words', False)\n        self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)\n        self.block_other_scripts_var = self.config.get('block_other_scripts', False)\n\n")], 'src/settings_persistence.py': [("            ('bias_repetitive_phrases', ['bias_repetitive_phrases_var'], False, bool),\n\n            # Glossary anti-duplicate parameters (separate from translation)\n", "            ('bias_repetitive_phrases', ['bias_repetitive_phrases_var'], False, bool),\n            ('block_other_scripts', ['block_other_scripts_var'], False, bool),\n\n            # Glossary anti-duplicate parameters (separate from translation)\n")], 'src/run_env.py': [("            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n", "            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var else '0',\n            'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n"), ("            'BIAS_REPETITIVE_PHRASES': 'Bias against repetitive phrases',\n            \n            # Azure\n", "            'BIAS_REPETITIVE_PHRASES': 'Bias against repetitive phrases',\n            'BLOCK_OTHER_SCRIPTS': 'Block other scripts in translations',\n            \n            # Azure\n"), ("                ('BIAS_REPETITIVE_PHRASES', '1' if getattr(self, 'bias_repetitive_phrases_var', False) else '0'),\n\n                # Azure API version\n", "                ('BIAS_REPETITIVE_PHRASES', '1' if getattr(self, 'bias_repetitive_phrases_var', False) else '0'),\n                ('BLOCK_OTHER_SCRIPTS', '1' if getattr(self, 'block_other_scripts_var', False) else '0'),\n\n                # Azure API version\n")]}
_BLOCK_OTHER_SCRIPTS_SCHEMA = {'src/settings_schema_data.py': [(" 'bias_repetitive_phrases',\n 'glossary_enable_anti_duplicate',\n", " 'bias_repetitive_phrases',\n 'block_other_scripts',\n 'glossary_enable_anti_duplicate',\n"), (" ('bias_repetitive_phrases', ('bias_repetitive_phrases_var',), ('value', False), ('bool',)),\n ('glossary_enable_anti_duplicate',\n", " ('bias_repetitive_phrases', ('bias_repetitive_phrases_var',), ('value', False), ('bool',)),\n ('block_other_scripts', ('block_other_scripts_var',), ('value', False), ('bool',)),\n ('glossary_enable_anti_duplicate',\n"), ("     'var_names': ('bias_repetitive_phrases_var',)},\n    'book_title_prompt': {'converter': ('str',),\n", '     \'var_names\': (\'bias_repetitive_phrases_var\',)},\n    \'block_other_scripts\': {\'converter\': (\'bool\',),\n     \'default\': False,\n     \'default_source\': \'init\',\n     \'dialog_default\': False,\n     \'env\': ((\'BLOCK_OTHER_SCRIPTS\', \'translation\', False),\n             (\'BLOCK_OTHER_SCRIPTS\', \'startup\', False)),\n     \'init_default\': False,\n     \'label\': \'Block other scripts in translations (adaptive, all models)\',\n     \'origins\': (\'settings_map\', \'init\', \'dialog\', \'env\'),\n     \'save_default\': False,\n     \'tooltip\': \'⚠️ Can cause errors or retries: mixed-language books, kept original names, \'\n                \'targets it does not recognise.\\n\'\n                \'Translations may only use the scripts of the target language: English output \'\n                \'blocks Chinese,\\n\'\n                "Japanese and Korean characters, Hindi, Thai, Cyrillic and the rest; the target\'s "\n                \'own script is\\n\'\n                \'never blocked, and glossary and single-pass glossary requests are left alone.\\n\'\n                \'Official OpenAI chat models (GPT-4o / GPT-4.1) also get a logit bias against the \'\n                \'most common\\n\'\n                \'tokens of those scripts. On every model, an answer with such text is sent again \'\n                \'with an\\n\'\n                \'instruction to write only in the target language, using the normal retry budget \'\n                \'(Max Retries).\\n\'\n                \'Works only while Enable Anti-Duplicate Parameters is on.\',\n     \'type\': \'bool\',\n     \'ui_group\': \'🎯 Anti-Duplicate Parameters\',\n     \'ui_sites\': (\'other.anti_duplicate\',),\n     \'ui_tab\': \'Logit Bias\',\n     \'var_names\': (\'block_other_scripts_var\',)},\n    \'book_title_prompt\': {\'converter\': (\'str\',),\n')]}

# 2026-10-28 "Allow emoticons and symbols from other scripts (kaomoji)" (script_ban_allow_kaomoji /
# SCRIPT_BAN_ALLOW_KAOMOJI, off): one row / env line after each block_other_scripts one, and the schema
# hunks of the current pin
_SCRIPT_BAN_ALLOW_KAOMOJI_LEGACY = {'src/translator_gui.py': [("        self.block_other_scripts_var = self.config.get('block_other_scripts', False)\n\n", "        self.block_other_scripts_var = self.config.get('block_other_scripts', False)\n        self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)\n\n"), ("                ('block_other_scripts', ['block_other_scripts_var'], False, bool),\n                \n", "                ('block_other_scripts', ['block_other_scripts_var'], False, bool),\n                ('script_ban_allow_kaomoji', ['script_ban_allow_kaomoji_var'], False, bool),\n                \n"), ("            'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n", "            'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n            'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n"), ("            'BLOCK_OTHER_SCRIPTS': 'Block other scripts in translations',\n            \n", "            'BLOCK_OTHER_SCRIPTS': 'Block other scripts in translations',\n            'SCRIPT_BAN_ALLOW_KAOMOJI': 'Allow emoticons from other scripts (kaomoji)',\n            \n"), ("                ('BLOCK_OTHER_SCRIPTS', '1' if getattr(self, 'block_other_scripts_var', False) else '0'),\n\n", "                ('BLOCK_OTHER_SCRIPTS', '1' if getattr(self, 'block_other_scripts_var', False) else '0'),\n                ('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0'),\n\n")], 'src/owner_state.py': [("        self.block_other_scripts_var = self.config.get('block_other_scripts', False)\n\n", "        self.block_other_scripts_var = self.config.get('block_other_scripts', False)\n        self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)\n\n")], 'src/settings_persistence.py': [("            ('block_other_scripts', ['block_other_scripts_var'], False, bool),\n\n", "            ('block_other_scripts', ['block_other_scripts_var'], False, bool),\n            ('script_ban_allow_kaomoji', ['script_ban_allow_kaomoji_var'], False, bool),\n\n")], 'src/run_env.py': [("            'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n", "            'BLOCK_OTHER_SCRIPTS': '1' if getattr(self, 'block_other_scripts_var', False) else '0',\n            'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',\n            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),\n"), ("            'BLOCK_OTHER_SCRIPTS': 'Block other scripts in translations',\n            \n", "            'BLOCK_OTHER_SCRIPTS': 'Block other scripts in translations',\n            'SCRIPT_BAN_ALLOW_KAOMOJI': 'Allow emoticons from other scripts (kaomoji)',\n            \n"), ("                ('BLOCK_OTHER_SCRIPTS', '1' if getattr(self, 'block_other_scripts_var', False) else '0'),\n\n", "                ('BLOCK_OTHER_SCRIPTS', '1' if getattr(self, 'block_other_scripts_var', False) else '0'),\n                ('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0'),\n\n")]}
_SCRIPT_BAN_ALLOW_KAOMOJI_SCHEMA = {'src/settings_schema_data.py': [(" 'block_other_scripts',\n 'glossary_enable_anti_duplicate',\n", " 'block_other_scripts',\n 'script_ban_allow_kaomoji',\n 'glossary_enable_anti_duplicate',\n"), (" ('block_other_scripts', ('block_other_scripts_var',), ('value', False), ('bool',)),\n ('glossary_enable_anti_duplicate',\n", " ('block_other_scripts', ('block_other_scripts_var',), ('value', False), ('bool',)),\n ('script_ban_allow_kaomoji', ('script_ban_allow_kaomoji_var',), ('value', False), ('bool',)),\n ('glossary_enable_anti_duplicate',\n"), ("     'var_names': ('scan_phase_mode_var',)},\n    'sdlxliff_one_column_layout': {'type': 'str', 'ui_sites': ('progress',)},\n", '     \'var_names\': (\'scan_phase_mode_var\',)},\n    \'script_ban_allow_kaomoji\': {\'converter\': (\'bool\',),\n     \'default\': False,\n     \'default_source\': \'init\',\n     \'dialog_default\': False,\n     \'env\': ((\'SCRIPT_BAN_ALLOW_KAOMOJI\', \'translation\', False),\n             (\'SCRIPT_BAN_ALLOW_KAOMOJI\', \'startup\', False)),\n     \'init_default\': False,\n     \'label\': \'Allow emoticons and symbols from other scripts (kaomoji)\',\n     \'origins\': (\'settings_map\', \'init\', \'dialog\', \'env\'),\n     \'save_default\': False,\n     \'tooltip\': \'Lets kaomoji through the script ban: ¯\\\\_(ツ)_/¯, (ノ°▽°)ノ, ಠ_ಠ, ٩(◕‿◕)۶, (╯°□°）╯︵ \'\n                \'┻━┻.\\n\'\n                \'Chinese / Japanese punctuation and brackets that carry no language (「」『』・｡、〜 ︵ ）) \'\n                \'are never\\n\'\n                \'blocked, and characters of other scripts that the source itself uses in emoticons \'\n                \'can be copied.\\n\'\n                \'Letting these characters through means a little more foreign text can slip by.\\n\'\n                "Applies to Block other scripts in translations and to the on-device engine\'s own "\n                \'switch.\',\n     \'type\': \'bool\',\n     \'ui_group\': \'🎯 Anti-Duplicate Parameters\',\n     \'ui_sites\': (\'other.anti_duplicate\',),\n     \'ui_tab\': \'Logit Bias\',\n     \'var_names\': (\'script_ban_allow_kaomoji_var\',)},\n    \'sdlxliff_one_column_layout\': {\'type\': \'str\', \'ui_sites\': (\'progress\',)},\n')]}


# 2026-10-30 "Whitelist source text" (script_ban_whitelist_source / SCRIPT_BAN_WHITELIST_SOURCE, off): one row /
# env line after each script_ban_allow_kaomoji one, and the schema hunks
_SCRIPT_BAN_WHITELIST_LEGACY = {'src/translator_gui.py': [("        self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)\n", "        self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)\n        self.script_ban_whitelist_source_var = self.config.get('script_ban_whitelist_source', False)\n"), ("                ('script_ban_allow_kaomoji', ['script_ban_allow_kaomoji_var'], False, bool),\n", "                ('script_ban_allow_kaomoji', ['script_ban_allow_kaomoji_var'], False, bool),\n                ('script_ban_whitelist_source', ['script_ban_whitelist_source_var'], False, bool),\n"), ("            'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',\n", "            'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',\n            'SCRIPT_BAN_WHITELIST_SOURCE': '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0',\n"), ("            'SCRIPT_BAN_ALLOW_KAOMOJI': 'Allow emoticons from other scripts (kaomoji)',\n", "            'SCRIPT_BAN_ALLOW_KAOMOJI': 'Allow emoticons from other scripts (kaomoji)',\n            'SCRIPT_BAN_WHITELIST_SOURCE': 'Whitelist source text',\n"), ("                ('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0'),\n", "                ('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0'),\n                ('SCRIPT_BAN_WHITELIST_SOURCE', '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0'),\n")], 'src/owner_state.py': [("        self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)\n", "        self.script_ban_allow_kaomoji_var = self.config.get('script_ban_allow_kaomoji', False)\n        self.script_ban_whitelist_source_var = self.config.get('script_ban_whitelist_source', False)\n")], 'src/settings_persistence.py': [("            ('script_ban_allow_kaomoji', ['script_ban_allow_kaomoji_var'], False, bool),\n", "            ('script_ban_allow_kaomoji', ['script_ban_allow_kaomoji_var'], False, bool),\n            ('script_ban_whitelist_source', ['script_ban_whitelist_source_var'], False, bool),\n")], 'src/run_env.py': [("            'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',\n", "            'SCRIPT_BAN_ALLOW_KAOMOJI': '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0',\n            'SCRIPT_BAN_WHITELIST_SOURCE': '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0',\n"), ("            'SCRIPT_BAN_ALLOW_KAOMOJI': 'Allow emoticons from other scripts (kaomoji)',\n", "            'SCRIPT_BAN_ALLOW_KAOMOJI': 'Allow emoticons from other scripts (kaomoji)',\n            'SCRIPT_BAN_WHITELIST_SOURCE': 'Whitelist source text',\n"), ("                ('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0'),\n", "                ('SCRIPT_BAN_ALLOW_KAOMOJI', '1' if getattr(self, 'script_ban_allow_kaomoji_var', False) else '0'),\n                ('SCRIPT_BAN_WHITELIST_SOURCE', '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0'),\n")]}
_SCRIPT_BAN_WHITELIST_SCHEMA = {'src/settings_schema_data.py': [(" 'script_ban_allow_kaomoji',\n 'glossary_enable_anti_duplicate',\n", " 'script_ban_allow_kaomoji',\n 'script_ban_whitelist_source',\n 'glossary_enable_anti_duplicate',\n"), (" ('script_ban_allow_kaomoji', ('script_ban_allow_kaomoji_var',), ('value', False), ('bool',)),\n", " ('script_ban_allow_kaomoji', ('script_ban_allow_kaomoji_var',), ('value', False), ('bool',)),\n ('script_ban_whitelist_source', ('script_ban_whitelist_source_var',), ('value', False), ('bool',)),\n"), ("    'sdlxliff_one_column_layout': {'type': 'str', 'ui_sites': ('progress',)},\n", '    \'script_ban_whitelist_source\': {\'converter\': (\'bool\',),\n     \'default\': False,\n     \'default_source\': \'init\',\n     \'dialog_default\': False,\n     \'env\': ((\'SCRIPT_BAN_WHITELIST_SOURCE\', \'translation\', False),\n             (\'SCRIPT_BAN_WHITELIST_SOURCE\', \'startup\', False)),\n     \'init_default\': False,\n     \'label\': \'Whitelist source text\',\n     \'origins\': (\'settings_map\', \'init\', \'dialog\', \'env\'),\n     \'save_default\': False,\n     \'tooltip\': \'Every request blocked by Block other scripts may also use the characters of its \'\n                \'own source text,\\n\'\n                \'so original names can be kept. Characters that are not in the source stay \'\n                \'blocked.\\n\'\n                \'This also lets a model copy the source instead of translating it.\\n\'\n                "Applies to Block other scripts in translations and to the on-device engine\'s own "\n                \'switch.\',\n     \'type\': \'bool\',\n     \'ui_group\': \'🎯 Anti-Duplicate Parameters\',\n     \'ui_sites\': (\'other.anti_duplicate\',),\n     \'ui_tab\': \'Logit Bias\',\n     \'var_names\': (\'script_ban_whitelist_source_var\',)},\n    \'sdlxliff_one_column_layout\': {\'type\': \'str\', \'ui_sites\': (\'progress\',)},\n')]}


# 2026-10-30 "Forced Truncation" (forced_truncation_enabled, off / forced_truncation_seconds, 3; FORCED_TRUNCATION,
# FORCED_TRUNCATION_SECONDS): rows / env lines after each script_ban_whitelist_source one, and the schema hunks
_FORCED_TRUNCATION_LEGACY = {'src/translator_gui.py': [("        self.script_ban_whitelist_source_var = self.config.get('script_ban_whitelist_source', False)\n", "        self.script_ban_whitelist_source_var = self.config.get('script_ban_whitelist_source', False)\n        self.forced_truncation_enabled_var = self.config.get('forced_truncation_enabled', False)\n        self.forced_truncation_seconds_var = self.config.get('forced_truncation_seconds', 3)\n"), ("                ('script_ban_whitelist_source', ['script_ban_whitelist_source_var'], False, bool),\n", "                ('script_ban_whitelist_source', ['script_ban_whitelist_source_var'], False, bool),\n                ('forced_truncation_enabled', ['forced_truncation_enabled_var'], False, bool),\n                ('forced_truncation_seconds', ['forced_truncation_seconds_var'], 3, int),\n"), ("            'SCRIPT_BAN_WHITELIST_SOURCE': '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0',\n", "            'SCRIPT_BAN_WHITELIST_SOURCE': '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0',\n            'FORCED_TRUNCATION': '1' if getattr(self, 'forced_truncation_enabled_var', False) else '0',\n            'FORCED_TRUNCATION_SECONDS': str(getattr(self, 'forced_truncation_seconds_var', 3)),\n"), ("            'SCRIPT_BAN_WHITELIST_SOURCE': 'Whitelist source text',\n", "            'SCRIPT_BAN_WHITELIST_SOURCE': 'Whitelist source text',\n            'FORCED_TRUNCATION': 'Forced Truncation',\n            'FORCED_TRUNCATION_SECONDS': 'Forced Truncation seconds',\n"), ("                ('SCRIPT_BAN_WHITELIST_SOURCE', '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0'),\n", "                ('SCRIPT_BAN_WHITELIST_SOURCE', '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0'),\n                ('FORCED_TRUNCATION', '1' if getattr(self, 'forced_truncation_enabled_var', False) else '0'),\n                ('FORCED_TRUNCATION_SECONDS', str(getattr(self, 'forced_truncation_seconds_var', 3))),\n")], 'src/owner_state.py': [("        self.script_ban_whitelist_source_var = self.config.get('script_ban_whitelist_source', False)\n", "        self.script_ban_whitelist_source_var = self.config.get('script_ban_whitelist_source', False)\n        self.forced_truncation_enabled_var = self.config.get('forced_truncation_enabled', False)\n        self.forced_truncation_seconds_var = self.config.get('forced_truncation_seconds', 3)\n")], 'src/settings_persistence.py': [("            ('script_ban_whitelist_source', ['script_ban_whitelist_source_var'], False, bool),\n", "            ('script_ban_whitelist_source', ['script_ban_whitelist_source_var'], False, bool),\n            ('forced_truncation_enabled', ['forced_truncation_enabled_var'], False, bool),\n            ('forced_truncation_seconds', ['forced_truncation_seconds_var'], 3, int),\n")], 'src/run_env.py': [("            'SCRIPT_BAN_WHITELIST_SOURCE': '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0',\n", "            'SCRIPT_BAN_WHITELIST_SOURCE': '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0',\n            'FORCED_TRUNCATION': '1' if getattr(self, 'forced_truncation_enabled_var', False) else '0',\n            'FORCED_TRUNCATION_SECONDS': str(getattr(self, 'forced_truncation_seconds_var', 3)),\n"), ("            'SCRIPT_BAN_WHITELIST_SOURCE': 'Whitelist source text',\n", "            'SCRIPT_BAN_WHITELIST_SOURCE': 'Whitelist source text',\n            'FORCED_TRUNCATION': 'Forced Truncation',\n            'FORCED_TRUNCATION_SECONDS': 'Forced Truncation seconds',\n"), ("                ('SCRIPT_BAN_WHITELIST_SOURCE', '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0'),\n", "                ('SCRIPT_BAN_WHITELIST_SOURCE', '1' if getattr(self, 'script_ban_whitelist_source_var', False) else '0'),\n                ('FORCED_TRUNCATION', '1' if getattr(self, 'forced_truncation_enabled_var', False) else '0'),\n                ('FORCED_TRUNCATION_SECONDS', str(getattr(self, 'forced_truncation_seconds_var', 3))),\n")]}
_FORCED_TRUNCATION_SCHEMA = {'src/settings_schema_data.py': [(" 'script_ban_whitelist_source',\n 'glossary_enable_anti_duplicate',\n", " 'script_ban_whitelist_source',\n 'forced_truncation_enabled',\n 'forced_truncation_seconds',\n 'glossary_enable_anti_duplicate',\n"), (" ('script_ban_whitelist_source', ('script_ban_whitelist_source_var',), ('value', False), ('bool',)),\n", " ('script_ban_whitelist_source', ('script_ban_whitelist_source_var',), ('value', False), ('bool',)),\n ('forced_truncation_enabled', ('forced_truncation_enabled_var',), ('value', False), ('bool',)),\n ('forced_truncation_seconds', ('forced_truncation_seconds_var',), ('value', 3), ('int',)),\n"), ("    'frequency_penalty': {'converter': ('float',),\n", "    'forced_truncation_enabled': {'converter': ('bool',),\n     'default': False,\n     'default_source': 'init',\n     'dialog_default': False,\n     'env': (('FORCED_TRUNCATION', 'translation', False), ('FORCED_TRUNCATION', 'startup', False)),\n     'init_default': False,\n     'label': 'Forced Truncation',\n     'origins': ('settings_map', 'init', 'dialog', 'env'),\n     'save_default': False,\n     'tooltip': 'When a streamed answer keeps repeating the same text for this many seconds, it is '\n                'stopped and\\n'\n                'ends with finish reason length, so the Progress Manager marks the chapter '\n                'truncated and the run\\n'\n                'moves on. Text that the source itself repeats as often is left alone, like in the '\n                'QA scanner.\\n'\n                'Works on ollamapull/ and OpenAI-compatible endpoints. On-device models have their '\n                'own switch.',\n     'type': 'bool',\n     'ui_group': '🎯 Anti-Duplicate Parameters',\n     'ui_sites': ('other.anti_duplicate',),\n     'ui_tab': 'Logit Bias',\n     'var_names': ('forced_truncation_enabled_var',)},\n    'forced_truncation_seconds': {'converter': ('int',),\n     'default': 3,\n     'default_source': 'init',\n     'dialog_default': 3,\n     'env': (('FORCED_TRUNCATION_SECONDS', 'translation', 3),\n             ('FORCED_TRUNCATION_SECONDS', 'startup', 3)),\n     'init_default': 3,\n     'origins': ('settings_map', 'init', 'dialog', 'env'),\n     'save_default': 3,\n     'type': 'int',\n     'ui_group': '🎯 Anti-Duplicate Parameters',\n     'ui_sites': ('other.anti_duplicate',),\n     'ui_tab': 'Logit Bias',\n     'var_names': ('forced_truncation_seconds_var',)},\n    'frequency_penalty': {'converter': ('float',),\n")]}


# 2026-10-30 "Mark glossary forced truncation as complete" (mark_glossary_forced_truncation_complete, off; MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE): rows / env lines after each
# forced_truncation_seconds one, and the schema hunks
_MARK_GLOSSARY_COMPLETE_LEGACY = {'src/translator_gui.py': [("        self.forced_truncation_seconds_var = self.config.get('forced_truncation_seconds', 3)\n", "        self.forced_truncation_seconds_var = self.config.get('forced_truncation_seconds', 3)\n        self.mark_glossary_forced_truncation_complete_var = self.config.get('mark_glossary_forced_truncation_complete', False)\n"), ("                ('forced_truncation_seconds', ['forced_truncation_seconds_var'], 3, int),\n", "                ('forced_truncation_seconds', ['forced_truncation_seconds_var'], 3, int),\n                ('mark_glossary_forced_truncation_complete', ['mark_glossary_forced_truncation_complete_var'], False, bool),\n"), ("            'FORCED_TRUNCATION_SECONDS': str(getattr(self, 'forced_truncation_seconds_var', 3)),\n", "            'FORCED_TRUNCATION_SECONDS': str(getattr(self, 'forced_truncation_seconds_var', 3)),\n            'MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE': '1' if getattr(self, 'mark_glossary_forced_truncation_complete_var', False) else '0',\n"), ("            'FORCED_TRUNCATION_SECONDS': 'Forced Truncation seconds',\n", "            'FORCED_TRUNCATION_SECONDS': 'Forced Truncation seconds',\n            'MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE': 'Mark glossary forced truncation as complete',\n"), ("                ('FORCED_TRUNCATION_SECONDS', str(getattr(self, 'forced_truncation_seconds_var', 3))),\n", "                ('FORCED_TRUNCATION_SECONDS', str(getattr(self, 'forced_truncation_seconds_var', 3))),\n                ('MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE', '1' if getattr(self, 'mark_glossary_forced_truncation_complete_var', False) else '0'),\n")], 'src/owner_state.py': [("        self.forced_truncation_seconds_var = self.config.get('forced_truncation_seconds', 3)\n", "        self.forced_truncation_seconds_var = self.config.get('forced_truncation_seconds', 3)\n        self.mark_glossary_forced_truncation_complete_var = self.config.get('mark_glossary_forced_truncation_complete', False)\n")], 'src/settings_persistence.py': [("            ('forced_truncation_seconds', ['forced_truncation_seconds_var'], 3, int),\n", "            ('forced_truncation_seconds', ['forced_truncation_seconds_var'], 3, int),\n            ('mark_glossary_forced_truncation_complete', ['mark_glossary_forced_truncation_complete_var'], False, bool),\n")], 'src/run_env.py': [("            'FORCED_TRUNCATION_SECONDS': str(getattr(self, 'forced_truncation_seconds_var', 3)),\n", "            'FORCED_TRUNCATION_SECONDS': str(getattr(self, 'forced_truncation_seconds_var', 3)),\n            'MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE': '1' if getattr(self, 'mark_glossary_forced_truncation_complete_var', False) else '0',\n"), ("            'FORCED_TRUNCATION_SECONDS': 'Forced Truncation seconds',\n", "            'FORCED_TRUNCATION_SECONDS': 'Forced Truncation seconds',\n            'MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE': 'Mark glossary forced truncation as complete',\n"), ("                ('FORCED_TRUNCATION_SECONDS', str(getattr(self, 'forced_truncation_seconds_var', 3))),\n", "                ('FORCED_TRUNCATION_SECONDS', str(getattr(self, 'forced_truncation_seconds_var', 3))),\n                ('MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE', '1' if getattr(self, 'mark_glossary_forced_truncation_complete_var', False) else '0'),\n")]}
_MARK_GLOSSARY_COMPLETE_SCHEMA = {'src/settings_schema_data.py': [(" 'forced_truncation_seconds',\n 'glossary_enable_anti_duplicate',\n", " 'forced_truncation_seconds',\n 'mark_glossary_forced_truncation_complete',\n 'glossary_enable_anti_duplicate',\n"), (" ('forced_truncation_seconds', ('forced_truncation_seconds_var',), ('value', 3), ('int',)),\n", " ('forced_truncation_seconds', ('forced_truncation_seconds_var',), ('value', 3), ('int',)),\n ('mark_glossary_forced_truncation_complete',\n"), ("    'max_image_dimension': {'default': 2048,\n", "    'mark_glossary_forced_truncation_complete': {'converter': ('bool',),\n     'default': False,\n     'default_source': 'init',\n     'dialog_default': False,\n     'env': (('MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE', 'translation', False),\n             ('MARK_GLOSSARY_FORCED_TRUNCATION_COMPLETE', 'startup', False)),\n     'init_default': False,\n     'label': 'Mark glossary forced truncation as complete',\n     'origins': ('settings_map', 'init', 'dialog', 'env'),\n     'save_default': False,\n     'tooltip': 'A glossary request stopped by Forced Truncation counts as finished: its chapter '\n                'is marked complete\\n'\n                'with the entries it has, instead of truncated, so it is not extracted again.\\n'\n                'Entries after the point where the model began repeating are never extracted.',\n     'type': 'bool',\n     'ui_group': '🎯 Anti-Duplicate Parameters',\n     'ui_sites': ('other.anti_duplicate',),\n     'ui_tab': 'Logit Bias',\n     'var_names': ('mark_glossary_forced_truncation_complete_var',)},\n    'max_image_dimension': {'default': 2048,\n")]}


# 2026-11-01 the QA scanner stops at once (the real diff of scan_html_folder.py)
_QA_STOP_SCANNER = [
    (
        'from functools import lru_cache\n'
        'import concurrent.futures\n'
        'import multiprocessing\n'
        'from threading import Lock\n'
    ,
        'from functools import lru_cache\n'
        'import concurrent.futures\n'
        'import contextlib\n'
        'import functools\n'
        'import multiprocessing\n'
        'from threading import Lock\n'
    ),
    (
        '    print("🛑 STOP SCAN CALLED - Global flag set to True")  # More visible debug\n'
        '    return True  # Return True to confirm it was called\n'
        '\n'
        'def _is_vision_output_mode(qa_settings=None):\n'
    ,
        '    print("🛑 STOP SCAN CALLED - Global flag set to True")  # More visible debug\n'
        '    return True  # Return True to confirm it was called\n'
        '\n'
        '\n'
        'class _ScanStopped(BaseException):\n'
        '    """Unwinds a stopped scan from wherever its thread is (``_raise_if_stopped``). A BaseException: the scan\'s\n'
        '    many ``except Exception`` blocks must not swallow it."""\n'
        '\n'
        '\n'
        '#: the stop check of the scan running on this thread (none on worker threads and outside a scan)\n'
        '_scan_thread = threading.local()\n'
        '\n'
        '\n'
        'def _raise_if_stopped():\n'
        '    """For the scan\'s long sequential work: ends the scan at once when a stop was requested."""\n'
        "    check = getattr(_scan_thread, 'should_stop', None)\n"
        '    if check is not None and check():\n'
        '        raise _ScanStopped()\n'
        '\n'
        '\n'
        'def _stops_at_once(scan):\n'
        '    """``scan_html_folder`` returns None as soon as ``_raise_if_stopped`` finds a stop, like its other stop paths."""\n'
        '    @functools.wraps(scan)\n'
        '    def wrapper(*args, **kwargs):\n'
        '        try:\n'
        '            return scan(*args, **kwargs)\n'
        '        except _ScanStopped:\n'
        '            return None\n'
        '        finally:\n'
        '            _scan_thread.should_stop = None\n'
        '    return wrapper\n'
        '\n'
        '\n'
        'def _abandon_pool(pool):\n'
        '    """Leave a pool without waiting for it: what is queued never starts and a process pool\'s workers are killed\n'
        '    (a thread worker cannot be, it sees the stop at its next file: ``_submit_worker``)."""\n'
        "    processes = list((getattr(pool, '_processes', None) or {}).values())\n"
        '    pool.shutdown(wait=False, cancel_futures=True)\n'
        '    for process in processes:\n'
        '        try:\n'
        '            process.kill()\n'
        '        except Exception:\n'
        '            pass\n'
        '\n'
        '\n'
        '@contextlib.contextmanager\n'
        'def _stoppable_pool(factory, should_stop, *args, **kwargs):\n'
        '    """``with factory(...) as pool``, except that leaving it after a stop does not wait (``_abandon_pool``)."""\n'
        '    pool = factory(*args, **kwargs)\n'
        '    try:\n'
        '        yield pool\n'
        '    finally:\n'
        '        if should_stop():\n'
        '            _abandon_pool(pool)\n'
        '        else:\n'
        '            pool.shutdown(wait=True)\n'
        '\n'
        '\n'
        'def _completed_until_stop(futures, should_stop, poll=0.1):\n'
        '    """``as_completed`` that notices a stop while it waits: it just ends."""\n'
        '    pending = set(futures)\n'
        '    while pending:\n'
        '        if should_stop():\n'
        '            return\n'
        '        done, pending = concurrent.futures.wait(pending, timeout=poll,\n'
        '                                                return_when=concurrent.futures.FIRST_COMPLETED)\n'
        '        yield from done\n'
        '\n'
        '\n'
        'def _submit_worker(pool, worker, args, should_stop):\n'
        '    """A thread worker gets the scan\'s stop check, for between its files. A process worker cannot see it, and does\n'
        '    not need to: its pool is killed."""\n'
        '    if isinstance(pool, concurrent.futures.ThreadPoolExecutor):\n'
        '        return pool.submit(worker, args, should_stop)\n'
        '    return pool.submit(worker, args)\n'
        '\n'
        '\n'
        'def _finish_api_calls_in_flight(pool, futures, should_stop, force_stopped, log, *, poll=0.1, force_grace=1.0):\n'
        '    """A stop during an API stage: the queued calls never start. A graceful stop lets the calls in flight finish;\n'
        '    a forced one (the second Stop) aborts them and does not wait for them beyond ``force_grace`` seconds. Polling\n'
        '    the stop check here is what notices that second Stop."""\n'
        '    pool.shutdown(wait=False, cancel_futures=True)\n'
        '    pending = {future for future in futures if not future.done()}\n'
        '    told = False\n'
        '    forced_at = None\n'
        '    while pending:\n'
        '        _done, pending = concurrent.futures.wait(pending, timeout=poll)\n'
        '        if not pending:\n'
        '            return\n'
        '        should_stop()\n'
        '        if forced_at is None and force_stopped():\n'
        '            forced_at = time.monotonic()\n'
        '        if forced_at is not None:\n'
        '            if time.monotonic() - forced_at >= force_grace:\n'
        '                return\n'
        '        elif not told:\n'
        '            told = True\n'
        '            log(f"   ⏳ Waiting for {len(pending)} API call(s) in flight to finish. Stop again to cancel them.")\n'
        '\n'
        'def _is_vision_output_mode(qa_settings=None):\n'
    ),
    (
        '            \n'
        '            # Step 2: Process files in spine order\n'
        '            if spine_files:\n'
        '                # Determine the base directory from content.opf location\n'
        "                base_dir = ''\n"
        '                if content_opf_path:\n'
        '                    # content.opf is often in OEBPS/ or similar\n'
        '                    base_dir = os.path.dirname(content_opf_path)\n'
        '                    if base_dir:\n'
        "                        base_dir = base_dir + '/'\n"
        '                \n'
        '                extracted_count = 0\n'
        '                for spine_index, file_info in sorted(spine_files.items()):\n'
        '                    try:\n'
        "                        file_path = file_info['href']\n"
        '                        \n'
        '                        # Try multiple path resolutions\n'
        '                        possible_paths = [\n'
        '                            file_path,                          # As-is from manifest\n'
        '                            base_dir + file_path,               # Relative to content.opf\n'
        "                            'OEBPS/' + file_path,               # Common EPUB structure\n"
        "                            'OPS/' + file_path,                 # Alternative structure\n"
        '                        ]\n'
        '                        \n'
        '                        content = None\n'
        '                        for try_path in possible_paths:\n'
    ,
        '            \n'
        '            # Step 2: Process files in spine order\n'
        '            if spine_files:\n'
        '                # Determine the base directory from content.opf location\n'
        "                base_dir = ''\n"
        '                if content_opf_path:\n'
        '                    # content.opf is often in OEBPS/ or similar\n'
        '                    base_dir = os.path.dirname(content_opf_path)\n'
        '                    if base_dir:\n'
        "                        base_dir = base_dir + '/'\n"
        '                \n'
        '                extracted_count = 0\n'
        '                for spine_index, file_info in sorted(spine_files.items()):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        file_path = file_info['href']\n"
        '                        \n'
        '                        # Try multiple path resolutions\n'
        '                        possible_paths = [\n'
        '                            file_path,                          # As-is from manifest\n'
        '                            base_dir + file_path,               # Relative to content.opf\n'
        "                            'OEBPS/' + file_path,               # Common EPUB structure\n"
        "                            'OPS/' + file_path,                 # Alternative structure\n"
        '                        ]\n'
        '                        \n'
        '                        content = None\n'
        '                        for try_path in possible_paths:\n'
    ),
    (
        '                \n'
        '                html_files.sort()\n'
        '                \n'
        '                for idx, file_path in enumerate(html_files, start=1):\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
        "                        soup = BeautifulSoup(content, 'html.parser')\n"
        "                        images = soup.find_all('img')\n"
    ,
        '                \n'
        '                html_files.sort()\n'
        '                \n'
        '                for idx, file_path in enumerate(html_files, start=1):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
        "                        soup = BeautifulSoup(content, 'html.parser')\n"
        "                        images = soup.find_all('img')\n"
    ),
    (
        '                    base_dir = os.path.dirname(content_opf_path)\n'
        '                    if base_dir:\n'
        "                        base_dir = base_dir + '/'\n"
        '                \n'
        '                extracted_count = 0\n'
        '                for spine_index, file_info in sorted(spine_files.items()):\n'
        '                    try:\n'
        "                        file_path = file_info['href']\n"
        '                        \n'
        '                        # Try multiple path resolutions\n'
        '                        possible_paths = [\n'
        '                            file_path,\n'
    ,
        '                    base_dir = os.path.dirname(content_opf_path)\n'
        '                    if base_dir:\n'
        "                        base_dir = base_dir + '/'\n"
        '                \n'
        '                extracted_count = 0\n'
        '                for spine_index, file_info in sorted(spine_files.items()):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        file_path = file_info['href']\n"
        '                        \n'
        '                        # Try multiple path resolutions\n'
        '                        possible_paths = [\n'
        '                            file_path,\n'
    ),
    (
        '                \n'
        '                html_files.sort()\n'
        '                \n'
        '                for idx, file_path in enumerate(html_files, start=1):\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
        "                        soup = BeautifulSoup(content, 'html.parser')\n"
        '                        # Remove non-visible tags before extracting text\n'
    ,
        '                \n'
        '                html_files.sort()\n'
        '                \n'
        '                for idx, file_path in enumerate(html_files, start=1):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
        "                        soup = BeautifulSoup(content, 'html.parser')\n"
        '                        # Remove non-visible tags before extracting text\n'
    ),
    (
        '\n'
        '                for spine_index, file_path in sorted(spine_files.items()):\n'
        '                    content = None\n'
        "                    for candidate in (file_path, base_dir + file_path, 'OEBPS/' + file_path, 'OPS/' + file_path):\n"
    ,
        '\n'
        '                for spine_index, file_path in sorted(spine_files.items()):\n'
        '                    _raise_if_stopped()\n'
        '                    content = None\n'
        "                    for candidate in (file_path, base_dir + file_path, 'OEBPS/' + file_path, 'OPS/' + file_path):\n"
    ),
    (
        '                ]\n'
        '                for index, file_path in enumerate(sorted(html_files), start=1):\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
    ,
        '                ]\n'
        '                for index, file_path in enumerate(sorted(html_files), start=1):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
    ),
    (
        '\n'
        '\n'
        'def process_enhance_duplicate_batch(args):\n'
        '    """Process a batch of enhanced duplicate detection - MUST BE AT MODULE LEVEL"""\n'
        '    batch_type, batch_data, worker_data = args\n'
    ,
        '\n'
        '\n'
        'def process_enhance_duplicate_batch(args, should_stop=None):\n'
        '    """Process a batch of enhanced duplicate detection - MUST BE AT MODULE LEVEL"""\n'
        '    batch_type, batch_data, worker_data = args\n'
    ),
    (
        '        \n'
        '        for idx1, idx2, file1, file2, chapter_num in comparisons:\n'
        '            # Get text data\n'
        '            data1 = text_data[idx1]\n'
    ,
        '        \n'
        '        for idx1, idx2, file1, file2, chapter_num in comparisons:\n'
        '            if should_stop is not None and should_stop():\n'
        '                break\n'
        '            # Get text data\n'
        '            data1 = text_data[idx1]\n'
    ),
    (
        'def enhance_duplicate_detection(results, duplicate_groups, duplicate_confidence, config, log, should_stop=None):\n'
        '    """Additional duplicate detection - PROCESSPOOLEXECUTOR VERSION"""\n'
        '\n'
        '    log("🔍 Enhanced duplicate detection (different naming formats)...")\n'
    ,
        'def enhance_duplicate_detection(results, duplicate_groups, duplicate_confidence, config, log, should_stop=None):\n'
        '    """Additional duplicate detection - PROCESSPOOLEXECUTOR VERSION"""\n'
        '    _stopped = should_stop or (lambda: False)\n'
        '\n'
        '    log("🔍 Enhanced duplicate detection (different naming formats)...")\n'
    ),
    (
        "            batches.append(('chapter_comparison', batch, worker_data))\n"
        '        \n'
        '        with _ExecCls(**_kw) as executor:\n'
        '            futures = []\n'
        '            \n'
        '            for batch_args in batches:\n'
        '                if should_stop and should_stop():\n'
        '                    log("⛔ Enhanced detection interrupted by user.")\n'
        '                    executor.shutdown(wait=True)\n'
        '                    return duplicates_found\n'
        '                \n'
        '                future = executor.submit(process_enhance_duplicate_batch, batch_args)\n'
        '                futures.append(future)\n'
        '            \n'
        '            # Collect results\n'
        '            for future in concurrent.futures.as_completed(futures):\n'
        '                batch_results = future.result()\n'
        '                \n'
    ,
        "            batches.append(('chapter_comparison', batch, worker_data))\n"
        '        \n'
        '        with _stoppable_pool(_ExecCls, _stopped, **_kw) as executor:\n'
        '            futures = []\n'
        '\n'
        '            for batch_args in batches:\n'
        '                if _stopped():\n'
        '                    log("⛔ Enhanced detection interrupted by user.")\n'
        '                    return duplicates_found\n'
        '\n'
        '                future = _submit_worker(executor, process_enhance_duplicate_batch, batch_args, _stopped)\n'
        '                futures.append(future)\n'
        '            \n'
        '            # Collect results\n'
        '            for future in _completed_until_stop(futures, _stopped):\n'
        '                batch_results = future.result()\n'
        '                \n'
    ),
    (
        "            batches.append(('preview_comparison', batch, worker_data))\n"
        '        \n'
        '        with _ExecCls(**_kw) as executor:\n'
        '            futures = []\n'
        '            \n'
        '            for batch_args in batches:\n'
        '                if should_stop and should_stop():\n'
        '                    log("⛔ Enhanced detection interrupted by user.")\n'
        '                    executor.shutdown(wait=True)\n'
        '                    return duplicates_found\n'
        '                \n'
        '                future = executor.submit(process_enhance_duplicate_batch, batch_args)\n'
        '                futures.append(future)\n'
        '            \n'
        '            # Collect results with progress\n'
        '            completed = 0\n'
        '            for future in concurrent.futures.as_completed(futures):\n'
        '                completed += 1\n'
        '                if completed % 10 == 0:\n'
    ,
        "            batches.append(('preview_comparison', batch, worker_data))\n"
        '        \n'
        '        with _stoppable_pool(_ExecCls, _stopped, **_kw) as executor:\n'
        '            futures = []\n'
        '\n'
        '            for batch_args in batches:\n'
        '                if _stopped():\n'
        '                    log("⛔ Enhanced detection interrupted by user.")\n'
        '                    return duplicates_found\n'
        '\n'
        '                future = _submit_worker(executor, process_enhance_duplicate_batch, batch_args, _stopped)\n'
        '                futures.append(future)\n'
        '            \n'
        '            # Collect results with progress\n'
        '            completed = 0\n'
        '            for future in _completed_until_stop(futures, _stopped):\n'
        '                completed += 1\n'
        '                if completed % 10 == 0:\n'
    ),
    (
        '    return duplicate_groups, near_duplicate_groups, duplicate_confidence\n'
        '\n'
        'def process_deep_similarity_batch(args):\n'
        '    """Process a batch of deep similarity comparisons with enhanced error handling"""\n'
        '    try:\n'
    ,
        '    return duplicate_groups, near_duplicate_groups, duplicate_confidence\n'
        '\n'
        'def process_deep_similarity_batch(args, should_stop=None):\n'
        '    """Process a batch of deep similarity comparisons with enhanced error handling"""\n'
        '    try:\n'
    ),
    (
        '        \n'
        '        for i, j, filename_i, filename_j in batch:\n'
        '            try:\n'
        '                # Get text samples\n'
    ,
        '        \n'
        '        for i, j, filename_i, filename_j in batch:\n'
        '            if should_stop is not None and should_stop():\n'
        '                break\n'
        '            try:\n'
        '                # Get text samples\n'
    ),
    (
        '        try:\n'
        '            # Process with ProcessPoolExecutor\n'
        '            with mobile_runtime.make_pool_executor(max_workers) as executor:\n'
        '                # Submit all batches\n'
        '                futures = []\n'
    ,
        '        try:\n'
        '            # Process with ProcessPoolExecutor\n'
        '            with _stoppable_pool(mobile_runtime.make_pool_executor, should_stop, max_workers) as executor:\n'
        '                # Submit all batches\n'
        '                futures = []\n'
    ),
    (
        '                    if should_stop():\n'
        '                        log("⛔ Deep similarity check interrupted by user.")\n'
        '                        executor.shutdown(wait=True)\n'
        '                        return\n'
        '                    \n'
        '                    future = executor.submit(process_deep_similarity_batch, args)\n'
        '                    futures.append(future)\n'
        '                \n'
        '                # Process results as they complete\n'
        '                for completed_future in concurrent.futures.as_completed(futures):\n'
        '                    if should_stop():\n'
        '                        log("⛔ Deep similarity check interrupted by user.")\n'
        '                        executor.shutdown(wait=True)\n'
        '                        return\n'
        '                    \n'
        '                    try:\n'
        '                        # NO TIMEOUT - let it run as long as needed\n'
    ,
        '                    if should_stop():\n'
        '                        log("⛔ Deep similarity check interrupted by user.")\n'
        '                        return\n'
        '\n'
        '                    future = _submit_worker(executor, process_deep_similarity_batch, args, should_stop)\n'
        '                    futures.append(future)\n'
        '\n'
        '                # Process results as they complete\n'
        '                for completed_future in _completed_until_stop(futures, should_stop):\n'
        '                    if should_stop():\n'
        '                        break\n'
        '\n'
        '                    try:\n'
        '                        # NO TIMEOUT - let it run as long as needed\n'
    ),
    (
        '                        break\n'
        '                \n'
        '                # If we completed successfully\n'
        '                if use_parallel:\n'
    ,
        '                        break\n'
        '                \n'
        '                if should_stop():\n'
        '                    log("⛔ Deep similarity check interrupted by user.")\n'
        '                    return\n'
        '\n'
        '                # If we completed successfully\n'
        '                if use_parallel:\n'
    ),
    (
        '            \n'
        '            # Step 2: Process files in spine order\n'
        '            if spine_files:\n'
        '                # Determine the base directory from content.opf location\n'
        "                base_dir = ''\n"
        '                if content_opf_path:\n'
        '                    # content.opf is often in OEBPS/ or similar\n'
        '                    base_dir = os.path.dirname(content_opf_path)\n'
        '                    if base_dir:\n'
        "                        base_dir = base_dir + '/'\n"
        '                \n'
        '                extracted_count = 0\n'
        '                for spine_index, file_info in sorted(spine_files.items()):\n'
        '                    try:\n'
        "                        file_path = file_info['href']\n"
        '                        \n'
        '                        # Try multiple path resolutions\n'
        '                        possible_paths = [\n'
        '                            file_path,                          # As-is from manifest\n'
        '                            base_dir + file_path,               # Relative to content.opf\n'
        "                            'OEBPS/' + file_path,               # Common EPUB structure\n"
        "                            'OPS/' + file_path,                 # Alternative structure\n"
        '                        ]\n'
        '                        \n'
        '                        content = None\n'
        '                        successful_path = None\n'
    ,
        '            \n'
        '            # Step 2: Process files in spine order\n'
        '            if spine_files:\n'
        '                # Determine the base directory from content.opf location\n'
        "                base_dir = ''\n"
        '                if content_opf_path:\n'
        '                    # content.opf is often in OEBPS/ or similar\n'
        '                    base_dir = os.path.dirname(content_opf_path)\n'
        '                    if base_dir:\n'
        "                        base_dir = base_dir + '/'\n"
        '                \n'
        '                extracted_count = 0\n'
        '                for spine_index, file_info in sorted(spine_files.items()):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        file_path = file_info['href']\n"
        '                        \n'
        '                        # Try multiple path resolutions\n'
        '                        possible_paths = [\n'
        '                            file_path,                          # As-is from manifest\n'
        '                            base_dir + file_path,               # Relative to content.opf\n'
        "                            'OEBPS/' + file_path,               # Common EPUB structure\n"
        "                            'OPS/' + file_path,                 # Alternative structure\n"
        '                        ]\n'
        '                        \n'
        '                        content = None\n'
        '                        successful_path = None\n'
    ),
    (
        '                \n'
        '                html_files.sort()\n'
        '                \n'
        '                for idx, file_path in enumerate(html_files, start=1):\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
        "                        soup = BeautifulSoup(content, 'html.parser')\n"
        '                        text = soup.get_text(strip=True)\n'
    ,
        '                \n'
        '                html_files.sort()\n'
        '                \n'
        '                for idx, file_path in enumerate(html_files, start=1):\n'
        '                    _raise_if_stopped()\n'
        '                    try:\n'
        "                        content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
        "                        soup = BeautifulSoup(content, 'html.parser')\n"
        '                        text = soup.get_text(strip=True)\n'
    ),
    (
        '\n'
        '        for idx, file_path in enumerate(html_files, start=1):\n'
        '            try:\n'
        "                content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
    ,
        '\n'
        '        for idx, file_path in enumerate(html_files, start=1):\n'
        '            _raise_if_stopped()\n'
        '            try:\n'
        "                content = zf.read(file_path).decode('utf-8', errors='ignore')\n"
    ),
    (
        '            spine_index = 1\n'
        "            for itemref in spine.find_all('itemref'):\n"
        "                idref = itemref.get('idref')\n"
        '                if idref and idref in manifest_map:\n'
    ,
        '            spine_index = 1\n'
        "            for itemref in spine.find_all('itemref'):\n"
        '                _raise_if_stopped()\n'
        "                idref = itemref.get('idref')\n"
        '                if idref and idref in manifest_map:\n'
    ),
    (
        '    }\n'
        '\n'
        'def process_html_file_batch(args):\n'
        '    """Process a batch of HTML or text files - MUST BE AT MODULE LEVEL"""\n'
        '    if len(args) >= 12:\n'
    ,
        '    }\n'
        '\n'
        'def process_html_file_batch(args, should_stop=None):\n'
        '    """Process a batch of HTML or text files - MUST BE AT MODULE LEVEL"""\n'
        '    if len(args) >= 12:\n'
    ),
    (
        '    \n'
        '    for idx, filename in file_batch:\n'
        '        full_path = os.path.join(folder_path, filename)\n'
        '        \n'
    ,
        '    \n'
        '    for idx, filename in file_batch:\n'
        '        if should_stop is not None and should_stop():\n'
        '            break\n'
        '        full_path = os.path.join(folder_path, filename)\n'
        '        \n'
    ),
    (
        '\n'
        '\n'
        "def scan_html_folder(folder_path, log=print, stop_flag=None, mode='quick-scan', qa_settings=None, epub_path=None, selected_files=None, text_file_mode=None, progress_path=None):\n"
        '    """\n'
    ,
        '\n'
        '\n'
        '@_stops_at_once\n'
        "def scan_html_folder(folder_path, log=print, stop_flag=None, mode='quick-scan', qa_settings=None, epub_path=None, selected_files=None, text_file_mode=None, progress_path=None):\n"
        '    """\n'
    ),
    (
        '                log("⛔ Graceful stop requested via GRACEFUL_STOP env var")\n'
        '            return True\n'
        '        return False\n'
        '    \n'
        '    start_time = time.time()\n'
        '    \n'
    ,
        '                log("⛔ Graceful stop requested via GRACEFUL_STOP env var")\n'
        '            return True\n'
        '        # a stop stays requested for this scan: the GUI clears the flags soon after it ended, while a worker it\n'
        '        # did not wait for may still be asking\n'
        '        return _stop_logged[0]\n'
        '\n'
        '    def force_stopped():\n'
        '        """A forced stop (the second Stop): what aborts API calls in flight. A graceful one lets them finish."""\n'
        '        if not should_stop():\n'
        '            return False\n'
        "        if os.environ.get('GRACEFUL_STOP') != '1':\n"
        '            return True\n'
        '        try:\n'
        '            from unified_api_client import UnifiedClient\n'
        '            return UnifiedClient.is_globally_cancelled()\n'
        '        except Exception:\n'
        '            return False\n'
        '\n'
        '    _scan_thread.should_stop = should_stop\n'
        '\n'
        '    start_time = time.time()\n'
        '    \n'
    ),
    (
        '                # Extract image counts from HTML files in word_count folder\n'
        "                chunk_files = [f for f in os.listdir(word_count_folder) if f.lower().endswith(('.html', '.htm', '.xhtml'))]\n"
        '                if chunk_files:\n'
        '                    for chunk_file in chunk_files:\n'
        '                        chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                        try:\n'
        "                            with open(chunk_path, 'r', encoding='utf-8') as f:\n"
        '                                chunk_html = f.read()\n'
    ,
        '                # Extract image counts from HTML files in word_count folder\n'
        "                chunk_files = [f for f in os.listdir(word_count_folder) if f.lower().endswith(('.html', '.htm', '.xhtml'))]\n"
        '                if chunk_files:\n'
        '                    for chunk_file in chunk_files:\n'
        '                        _raise_if_stopped()\n'
        '                        chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                        try:\n'
        "                            with open(chunk_path, 'r', encoding='utf-8') as f:\n"
        '                                chunk_html = f.read()\n'
    ),
    (
        '                log(f"❗ Extracting punctuation information from word_count folder")\n'
        "                chunk_files = [f for f in os.listdir(word_count_folder) if f.lower().endswith(('.html', '.htm', '.xhtml'))]\n"
        '                if chunk_files:\n'
        '                    for chunk_file in chunk_files:\n'
        '                        chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                        try:\n'
        "                            with open(chunk_path, 'r', encoding='utf-8') as f:\n"
        '                                chunk_html = f.read()\n'
    ,
        '                log(f"❗ Extracting punctuation information from word_count folder")\n'
        "                chunk_files = [f for f in os.listdir(word_count_folder) if f.lower().endswith(('.html', '.htm', '.xhtml'))]\n"
        '                if chunk_files:\n'
        '                    for chunk_file in chunk_files:\n'
        '                        _raise_if_stopped()\n'
        '                        chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                        try:\n'
        "                            with open(chunk_path, 'r', encoding='utf-8') as f:\n"
        '                                chunk_html = f.read()\n'
    ),
    (
        '                ]\n'
        '                for chunk_file in chunk_files:\n'
        '                    chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                    try:\n'
    ,
        '                ]\n'
        '                for chunk_file in chunk_files:\n'
        '                    _raise_if_stopped()\n'
        '                    chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                    try:\n'
    ),
    (
        "                chunk_files = [f for f in os.listdir(word_count_folder) if f.lower().endswith(('.html', '.htm', '.xhtml'))]\n"
        '                for chunk_file in chunk_files:\n'
        '                    chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                    try:\n'
    ,
        "                chunk_files = [f for f in os.listdir(word_count_folder) if f.lower().endswith(('.html', '.htm', '.xhtml'))]\n"
        '                for chunk_file in chunk_files:\n'
        '                    _raise_if_stopped()\n'
        '                    chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        '                    try:\n'
    ),
    (
        '                            original_word_counts = {}\n'
        '                            for chunk_file in chunk_files:\n'
        '                                chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        "                                with open(chunk_path, 'r', encoding='utf-8') as f:\n"
    ,
        '                            original_word_counts = {}\n'
        '                            for chunk_file in chunk_files:\n'
        '                                _raise_if_stopped()\n'
        '                                chunk_path = os.path.join(word_count_folder, chunk_file)\n'
        "                                with open(chunk_path, 'r', encoding='utf-8') as f:\n"
    ),
    (
        '    broken_batches = []  # batches a broken process pool lost (a worker died): run again on threads below\n'
        '    \n'
        '    with _ExecutorCls(**_exec_kwargs) as executor:\n'
        '        # Submit all batches\n'
        '        futures = []\n'
        '        future_args = {}\n'
        '        \n'
        '        for args in worker_args:\n'
        '            if should_stop():\n'
        '                log("⛔ QA scan interrupted before processing.")\n'
        '                executor.shutdown(wait=True)\n'
        '                return\n'
        '            \n'
        '            future = executor.submit(process_html_file_batch, args)\n'
        '            futures.append(future)\n'
        '            future_args[future] = args\n'
        '        \n'
        '        # Collect results as they complete\n'
        '        for completed_idx, future in enumerate(concurrent.futures.as_completed(futures)):\n'
        '            if should_stop():\n'
        '                log("⛔ QA scan interrupted during processing.")\n'
        '                executor.shutdown(wait=True)\n'
        '                return\n'
        '            \n'
        '            try:\n'
        '                batch_results = future.result()\n'
    ,
        '    broken_batches = []  # batches a broken process pool lost (a worker died): run again on threads below\n'
        '    \n'
        '    with _stoppable_pool(_ExecutorCls, should_stop, **_exec_kwargs) as executor:\n'
        '        # Submit all batches\n'
        '        futures = []\n'
        '        future_args = {}\n'
        '\n'
        '        for args in worker_args:\n'
        '            if should_stop():\n'
        '                log("⛔ QA scan interrupted before processing.")\n'
        '                return\n'
        '\n'
        '            future = _submit_worker(executor, process_html_file_batch, args, should_stop)\n'
        '            futures.append(future)\n'
        '            future_args[future] = args\n'
        '\n'
        '        # Collect results as they complete\n'
        '        for completed_idx, future in enumerate(_completed_until_stop(futures, should_stop)):\n'
        '            if should_stop():\n'
        '                break\n'
        '\n'
        '            try:\n'
        '                batch_results = future.result()\n'
    ),
    (
        '                import traceback\n'
        '                log(f"   Traceback: {traceback.format_exc()}")\n'
        '    \n'
        '    if broken_batches and not should_stop():\n'
        '        log(f"   🧵 Scanning the {len(broken_batches)} batch(es) it left with threads instead "\n'
        '            f"(its cause, if it crashed, is in crash.log in the logs folder)")\n'
        '        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:\n'
        '            for batch_results in thread_executor.map(process_html_file_batch, broken_batches):\n'
        '                if should_stop():\n'
        '                    log("⛔ QA scan interrupted during processing.")\n'
        '                    thread_executor.shutdown(wait=True, cancel_futures=True)\n'
        '                    return\n'
        '                processed_count += len(batch_results)\n'
        '                results.extend(batch_results)\n'
        '        log(f"   ✅ {processed_count}/{len(html_files)} files scanned")\n'
        '    \n'
    ,
        '                import traceback\n'
        '                log(f"   Traceback: {traceback.format_exc()}")\n'
        '\n'
        '        if should_stop():\n'
        '            log("⛔ QA scan interrupted during processing.")\n'
        '            return\n'
        '\n'
        '    if broken_batches and not should_stop():\n'
        '        log(f"   🧵 Scanning the {len(broken_batches)} batch(es) it left with threads instead "\n'
        '            f"(its cause, if it crashed, is in crash.log in the logs folder)")\n'
        '        with _stoppable_pool(concurrent.futures.ThreadPoolExecutor, should_stop, max_workers=max_workers) as thread_executor:\n'
        '            thread_futures = [_submit_worker(thread_executor, process_html_file_batch, batch, should_stop)\n'
        '                              for batch in broken_batches]\n'
        '            for thread_future in _completed_until_stop(thread_futures, should_stop):\n'
        '                if should_stop():\n'
        '                    break\n'
        '                batch_results = thread_future.result()\n'
        '                processed_count += len(batch_results)\n'
        '                results.extend(batch_results)\n'
        '            if should_stop():\n'
        '                log("⛔ QA scan interrupted during processing.")\n'
        '                return\n'
        '        log(f"   ✅ {processed_count}/{len(html_files)} files scanned")\n'
        '    \n'
    ),
    (
        '    \n'
        '    dup_time = time.time() - dup_start_time\n'
        '    log(f"✅ Duplicate detection completed in {dup_time:.1f} seconds")\n'
        '\n'
    ,
        '    \n'
        '    dup_time = time.time() - dup_start_time\n'
        '    if should_stop():\n'
        '        log("⛔ QA scan stopped during duplicate detection.")\n'
        '        return\n'
        '    log(f"✅ Duplicate detection completed in {dup_time:.1f} seconds")\n'
        '\n'
    ),
    (
        '            if not isinstance(source_info, dict):\n'
        '                continue\n'
        "            source_filename = os.path.basename(str(source_info.get('filename', ''))).lower()\n"
        '            source_basename = _strip_html_like_extensions(source_filename)\n'
    ,
        '            if not isinstance(source_info, dict):\n'
        '                continue\n'
        '            _raise_if_stopped()\n'
        "            source_filename = os.path.basename(str(source_info.get('filename', ''))).lower()\n"
        '            source_basename = _strip_html_like_extensions(source_filename)\n'
    ),
    (
        '        if should_stop():\n'
        '            log("⛔ QA scan stopped during issue processing.")\n'
        '            break\n'
        "        issues = result.get('issues', [])\n"
        '\n'
    ,
        '        if should_stop():\n'
        '            log("⛔ QA scan stopped during issue processing.")\n'
        '            return\n'
        "        issues = result.get('issues', [])\n"
        '\n'
    ),
    (
        '    if check_truncation and original_html_content and not should_stop():\n'
        '        import threading as _threading\n'
        '        from concurrent.futures import ThreadPoolExecutor, as_completed\n'
        '\n'
        "        src_lang = str(qa_settings.get('source_language', 'auto')).strip()\n"
    ,
        '    if check_truncation and original_html_content and not should_stop():\n'
        '        import threading as _threading\n'
        '        from concurrent.futures import ThreadPoolExecutor\n'
        '\n'
        "        src_lang = str(qa_settings.get('source_language', 'auto')).strip()\n"
    ),
    (
        '        log(f"\\n🔍 Running silent truncation detection ({max_workers} threads)...")\n'
        '\n'
        '        with ThreadPoolExecutor(max_workers=max_workers) as trunc_pool:\n'
        '            futures = {trunc_pool.submit(_check_one, r): r for r in results}\n'
        '            for future in as_completed(futures):\n'
        '                if should_stop():\n'
        '                    break\n'
    ,
        '        log(f"\\n🔍 Running silent truncation detection ({max_workers} threads)...")\n'
        '\n'
        '        with _stoppable_pool(ThreadPoolExecutor, should_stop, max_workers=max_workers) as trunc_pool:\n'
        '            futures = {trunc_pool.submit(_check_one, r): r for r in results}\n'
        '            for future in _completed_until_stop(futures, should_stop):\n'
        '                if should_stop():\n'
        '                    break\n'
    ),
    (
        '    if check_ai_truncation and original_html_content and not should_stop():\n'
        '        import threading as _threading_ai\n'
        '        from concurrent.futures import ThreadPoolExecutor as _AIThreadPool, as_completed as _ai_as_completed\n'
        '\n'
        "        _ai_tail_chars = int(qa_settings.get('ai_truncation_tail_chars', 400))\n"
    ,
        '    if check_ai_truncation and original_html_content and not should_stop():\n'
        '        import threading as _threading_ai\n'
        '        from concurrent.futures import ThreadPoolExecutor as _AIThreadPool\n'
        '\n'
        "        _ai_tail_chars = int(qa_settings.get('ai_truncation_tail_chars', 400))\n"
    ),
    (
        '            _ai_client = create_client_with_multi_key_support(_api_key, _model, _output_dir, _client_config)\n'
        "            _ai_client.context = 'qa_truncation'\n"
        "            # Wire QA scanner's stop flag into the client so in-flight API\n"
        '            # calls abort immediately when the user presses Stop.\n'
        '            # Force-stop only — no graceful stop needed for simple QA queries.\n'
        '            _ai_client._stop_callback = should_stop\n'
        '\n'
        "            _ai_uses_dedicated_qa_pool = bool(_ai_config.get('use_ai_truncation_detection_keys', False))\n"
    ,
        '            _ai_client = create_client_with_multi_key_support(_api_key, _model, _output_dir, _client_config)\n'
        "            _ai_client.context = 'qa_truncation'\n"
        "            # Wire QA scanner's force stop into the client so in-flight API calls\n"
        '            # abort immediately on it. A graceful stop lets them finish.\n'
        '            _ai_client._stop_callback = force_stopped\n'
        '\n'
        "            _ai_uses_dedicated_qa_pool = bool(_ai_config.get('use_ai_truncation_detection_keys', False))\n"
    ),
    (
        '                    worker_client = create_client_with_multi_key_support(_api_key, _model, _output_dir, _client_config)\n'
        "                    worker_client.context = 'qa_truncation'\n"
        '                    worker_client._stop_callback = should_stop\n'
        '                    _ai_worker_clients.client = worker_client\n'
        '                return worker_client\n'
    ,
        '                    worker_client = create_client_with_multi_key_support(_api_key, _model, _output_dir, _client_config)\n'
        "                    worker_client.context = 'qa_truncation'\n"
        '                    worker_client._stop_callback = force_stopped\n'
        '                    _ai_worker_clients.client = worker_client\n'
        '                return worker_client\n'
    ),
    (
        "                os.environ['BATCH_TRANSLATION'] = '1'\n"
        '            _ai_pool = None\n'
        '            try:\n'
        '                _ai_pool = _AIThreadPool(max_workers=_ai_max_workers)\n'
        '                _ai_futures = {_ai_pool.submit(_ai_check_one, r): r for r in results}\n'
        '                for _ai_future in _ai_as_completed(_ai_futures):\n'
        '                    if should_stop():\n'
        '                        for _pending in _ai_futures:\n'
        '                            _pending.cancel()\n'
        '                        break\n'
        '                    try:\n'
    ,
        "                os.environ['BATCH_TRANSLATION'] = '1'\n"
        '            _ai_pool = None\n'
        '            _ai_futures = {}\n'
        '            try:\n'
        '                _ai_pool = _AIThreadPool(max_workers=_ai_max_workers)\n'
        '                _ai_futures = {_ai_pool.submit(_ai_check_one, r): r for r in results}\n'
        '                for _ai_future in _completed_until_stop(_ai_futures, should_stop):\n'
        '                    if should_stop():\n'
        '                        break\n'
        '                    try:\n'
    ),
    (
        '            finally:\n'
        '                if _ai_pool is not None:\n'
        '                    try:\n'
        '                        # ALWAYS wait for threads to finish before returning.\n'
        '                        # cancel_futures=True kills QUEUED tasks instantly.\n'
        '                        # Running tasks hit stop checks in _ai_check_one/_setup_client\n'
        "                        # and exit quickly, so this won't block for long.\n"
        '                        # Using wait=False caused threads to outlive the scan function,\n'
        '                        # and _check_qa_stop_done would clear stop flags while threads\n'
        '                        # were still initializing clients.\n'
        '                        _ai_pool.shutdown(wait=True, cancel_futures=True)\n'
        '                    except TypeError:\n'
        '                        _ai_pool.shutdown(wait=True)\n'
        '                # Restore original BATCH_TRANSLATION env var\n'
    ,
        '            finally:\n'
        '                if _ai_pool is not None:\n'
        '                    if should_stop():\n'
        '                        _finish_api_calls_in_flight(_ai_pool, _ai_futures, should_stop, force_stopped, log)\n'
        '                    else:\n'
        '                        _ai_pool.shutdown(wait=True)\n'
        '                # Restore original BATCH_TRANSLATION env var\n'
    ),
    (
        '            if should_stop():\n'
        '                log(f"   ⛔ AI truncation detection stopped: {_ai_checked} checked, {_ai_flagged} flagged")\n'
        '            else:\n'
        '                log(f"   ✅ AI truncation detection complete: {_ai_checked} checked, {_ai_flagged} flagged")\n'
        '\n'
        "    if qa_settings.get('check_glossary_names', False) and not should_stop():\n"
    ,
        '            if should_stop():\n'
        '                log(f"   ⛔ AI truncation detection stopped: {_ai_checked} checked, {_ai_flagged} flagged")\n'
        '                return\n'
        '            log(f"   ✅ AI truncation detection complete: {_ai_checked} checked, {_ai_flagged} flagged")\n'
        '\n'
        "    if qa_settings.get('check_glossary_names', False) and not should_stop():\n"
    ),
    (
        '        except Exception as e:\n'
        '            log(f"   ⚠️ Glossary name check failed: {e}")\n'
        '\n'
        '    _attach_chunk_results_to_scan(\n'
    ,
        '        except Exception as e:\n'
        '            log(f"   ⚠️ Glossary name check failed: {e}")\n'
        '\n'
        '    if should_stop():\n'
        '        log("⛔ QA scan stopped before report generation.")\n'
        '        return\n'
        '\n'
        '    _attach_chunk_results_to_scan(\n'
    ),
    (
        "        result.pop('normalized_text', None)\n"
        "        result.pop('qa_header_text', None)\n"
        '    \n'
        '    if should_stop():\n'
        '        log("⛔ QA scan stopped before report generation.")\n'
        '        return\n'
        '\n'
        '    # Generate reports\n'
    ,
        "        result.pop('normalized_text', None)\n"
        "        result.pop('qa_header_text', None)\n"
        '\n'
        '    # Generate reports\n'
    ),
    (
        '# ADD THIS AT MODULE LEVEL (outside any function/class)\n'
        '\n'
        'def process_comparison_batch_fast(args):\n'
        '    """Process a batch of comparisons - MUST BE AT MODULE LEVEL FOR PICKLING"""\n'
        '    batch, data = args\n'
    ,
        '# ADD THIS AT MODULE LEVEL (outside any function/class)\n'
        '\n'
        'def process_comparison_batch_fast(args, should_stop=None):\n'
        '    """Process a batch of comparisons - MUST BE AT MODULE LEVEL FOR PICKLING"""\n'
        '    batch, data = args\n'
    ),
    (
        '    \n'
        '    for i, j in batch:\n'
        '        data_i = all_data[i]\n'
        '        data_j = all_data[j]\n'
    ,
        '    \n'
        '    for i, j in batch:\n'
        '        if should_stop is not None and should_stop():\n'
        '            break\n'
        '        data_i = all_data[i]\n'
        '        data_j = all_data[j]\n'
    ),
    (
        '    \n'
        '    # Process with ProcessPoolExecutor\n'
        '    with mobile_runtime.make_pool_executor(max_workers, initializer=_init_worker_process) as executor:\n'
        '        # Submit all batches\n'
        '        futures = []\n'
    ,
        '    \n'
        '    # Process with ProcessPoolExecutor\n'
        '    with _stoppable_pool(mobile_runtime.make_pool_executor, should_stop, max_workers,\n'
        '                         initializer=_init_worker_process) as executor:\n'
        '        # Submit all batches\n'
        '        futures = []\n'
    ),
    (
        '            if should_stop():\n'
        '                log("⛔ AI Hunter interrupted by user.")\n'
        '                executor.shutdown(wait=True)\n'
        '                return comparisons_done\n'
        '            \n'
        '            future = executor.submit(process_comparison_batch_fast, args)\n'
        '            futures.append(future)\n'
        '        \n'
        '        # Process results as they complete\n'
        '        for completed_future in concurrent.futures.as_completed(futures):\n'
        '            if should_stop():\n'
        '                log("⛔ AI Hunter interrupted by user.")\n'
        '                executor.shutdown(wait=True)\n'
        '                return comparisons_done\n'
        '            \n'
        '            # Get results\n'
        '            batch_results = completed_future.result()\n'
    ,
        '            if should_stop():\n'
        '                log("⛔ AI Hunter interrupted by user.")\n'
        '                return comparisons_done\n'
        '\n'
        '            future = _submit_worker(executor, process_comparison_batch_fast, args, should_stop)\n'
        '            futures.append(future)\n'
        '\n'
        '        # Process results as they complete\n'
        '        for completed_future in _completed_until_stop(futures, should_stop):\n'
        '            if should_stop():\n'
        '                log("⛔ AI Hunter interrupted by user.")\n'
        '                return comparisons_done\n'
        '\n'
        '            # Get results\n'
        '            batch_results = completed_future.result()\n'
    ),
    (
        '                last_progress = progress\n'
        '    \n'
        '    # Final summary\n'
        '    elapsed = time.time() - start_time\n'
    ,
        '                last_progress = progress\n'
        '    \n'
        '    if should_stop():\n'
        '        log("⛔ AI Hunter interrupted by user.")\n'
        '        return comparisons_done\n'
        '\n'
        '    # Final summary\n'
        '    elapsed = time.time() - start_time\n'
    ),
]

# 2026-11-01 "Fix Missing Tag Bracket" (fix_missing_tag_bracket / FIX_MISSING_TAG_BRACKET, off) and its QA check
# (check_missing_tag_bracket / QA_CHECK_MISSING_TAG_BRACKET, on)
_MISSING_BRACKET_LEGACY = {'src/translator_gui.py': [("                ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n                ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n", "                ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n                ('fix_missing_tag_bracket', ['fix_missing_tag_bracket_var'], False, bool),\n                ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n"), ("            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n", "            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('fix_missing_tag_bracket', ['fix_missing_tag_bracket_var'], False, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n"), ("            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            # Keep existing variables intact\n", "            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            ('fix_missing_tag_bracket_var', 'fix_missing_tag_bracket', False),\n            # Keep existing variables intact\n"), ("            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n", "            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            'FIX_MISSING_TAG_BRACKET': '1' if getattr(self, 'fix_missing_tag_bracket_var', False) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n"), ("                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n\n                # Post-translation scanning phase\n", "                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n                ('FIX_MISSING_TAG_BRACKET', '1' if getattr(self, 'fix_missing_tag_bracket_var', False) else '0'),\n\n                # Post-translation scanning phase\n"), ("        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n", "        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_missing_tag_bracket_var = self.config.get('fix_missing_tag_bracket', False)\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n"), ("        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n", "        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_MISSING_TAG_BRACKET'] = '1' if self.fix_missing_tag_bracket_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n")], 'src/owner_state.py': [("            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            # Keep existing variables intact\n", "            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            ('fix_missing_tag_bracket_var', 'fix_missing_tag_bracket', False),\n            # Keep existing variables intact\n"), ("        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n", "        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_missing_tag_bracket_var = self.config.get('fix_missing_tag_bracket', False)\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n"), ("        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n", "        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_MISSING_TAG_BRACKET'] = '1' if self.fix_missing_tag_bracket_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n")], 'src/settings_persistence.py': [("            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n", "            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('fix_missing_tag_bracket', ['fix_missing_tag_bracket_var'], False, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n")], 'src/run_env.py': [("            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n", "            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            'FIX_MISSING_TAG_BRACKET': '1' if getattr(self, 'fix_missing_tag_bracket_var', False) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n"), ("                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n\n                # Post-translation scanning phase\n", "                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n                ('FIX_MISSING_TAG_BRACKET', '1' if getattr(self, 'fix_missing_tag_bracket_var', False) else '0'),\n\n                # Post-translation scanning phase\n")]}
_MISSING_BRACKET_SCHEMA = {'src/settings_schema_data.py': [(" 'remove_excessive_empty_tags_on_device',\n 'remove_excessive_empty_tags_all_models',\n 'emergency_paragraph_restore',\n 'emergency_image_restore',\n", " 'remove_excessive_empty_tags_on_device',\n 'remove_excessive_empty_tags_all_models',\n 'fix_missing_tag_bracket',\n 'emergency_paragraph_restore',\n 'emergency_image_restore',\n"), ("  ('value', False),\n  ('bool',)),\n ('emergency_paragraph_restore', ('emergency_restore_var',), ('value', False), ('bool',)),\n ('emergency_image_restore', ('emergency_image_restore_var',), ('value', False), ('bool',)),\n", "  ('value', False),\n  ('bool',)),\n ('fix_missing_tag_bracket', ('fix_missing_tag_bracket_var',), ('value', False), ('bool',)),\n ('emergency_paragraph_restore', ('emergency_restore_var',), ('value', False), ('bool',)),\n ('emergency_image_restore', ('emergency_image_restore_var',), ('value', False), ('bool',)),\n"), ("  'remove_excessive_empty_tags_all_models',\n  ('value', False)),\n ('enable_image_translation_var', 'enable_image_translation', ('value', False)),\n ('process_webnovel_images_var', 'process_webnovel_images', ('value', True)),\n", "  'remove_excessive_empty_tags_all_models',\n  ('value', False)),\n ('fix_missing_tag_bracket_var', 'fix_missing_tag_bracket', ('value', False)),\n ('enable_image_translation_var', 'enable_image_translation', ('value', False)),\n ('process_webnovel_images_var', 'process_webnovel_images', ('value', True)),\n"), ("     'ui_group': 'Processing Options',\n     'ui_sites': ('other.processing',)},\n    'fix_stray_p_gt_bs': {'converter': ('bool',),\n     'default': False,\n", "     'ui_group': 'Processing Options',\n     'ui_sites': ('other.processing',)},\n    'fix_missing_tag_bracket': {'converter': ('bool',),\n     'default': False,\n     'default_source': 'init',\n     'dialog_default': False,\n     'env': (('FIX_MISSING_TAG_BRACKET', 'translation', False),\n             ('FIX_MISSING_TAG_BRACKET', 'startup', '<none>'),\n             ('FIX_MISSING_TAG_BRACKET', 'startup', False)),\n     'init_default': False,\n     'label': 'Fix Missing Tag Bracket - LLM Token Fix',\n     'origins': ('settings_map', 'init', 'dialog', 'env'),\n     'save_default': False,\n     'type': 'bool',\n     'ui_group': 'Processing Options',\n     'ui_sites': ('other.processing',),\n     'var_names': ('fix_missing_tag_bracket_var',)},\n    'fix_stray_p_gt_bs': {'converter': ('bool',),\n     'default': False,\n"), ("     'ui_group': 'Word Count Analysis',\n     'ui_sites': ('qa',)},\n    'qa_scanner_settings.check_multiple_headers': {'default': True,\n     'default_source': 'nested',\n", '     \'ui_group\': \'Word Count Analysis\',\n     \'ui_sites\': (\'qa\',)},\n    \'qa_scanner_settings.check_missing_tag_bracket\': {\'default\': True,\n     \'default_source\': \'nested\',\n     \'dialog_default\': True,\n     \'env\': ((\'QA_CHECK_MISSING_TAG_BRACKET\', \'qa\', True),),\n     \'label\': \'Check for missing tag brackets (<p"Hello"</p>)\',\n     \'nested_default\': True,\n     \'origins\': (\'nested\', \'dialog\', \'env\'),\n     \'parent\': \'qa_scanner_settings\',\n     \'type\': \'bool\',\n     \'ui_group\': \'Additional Checks\',\n     \'ui_sites\': (\'qa\',)},\n    \'qa_scanner_settings.check_multiple_headers\': {\'default\': True,\n     \'default_source\': \'nested\',\n')]}
_MISSING_BRACKET_QA = {'src/QA_Scanner_GUI.py': [("                    'sdlxliff_min_source_paragraph_tags': 20,\n                    'check_invalid_nesting': False,\n                    'cache_enabled': True,\n                    'cache_auto_size': False,\n", "                    'sdlxliff_min_source_paragraph_tags': 20,\n                    'check_invalid_nesting': False,\n                    'check_missing_tag_bracket': True,\n                    'cache_enabled': True,\n                    'cache_auto_size': False,\n"), ('            additional_layout.addWidget(check_invalid_nesting_checkbox)\n\n            # Silent truncation detection (optional, requires heavy deps)\n            check_truncation_checkbox = self._create_styled_checkbox("Silent truncation detection (compares tail paragraphs via back-translation + embeddings)")\n', '            additional_layout.addWidget(check_invalid_nesting_checkbox)\n\n            # Missing tag bracket check (LLM token issue, separate toggle)\n            check_missing_tag_bracket_checkbox = self._create_styled_checkbox("Check for missing tag brackets (<p\\"Hello\\"</p>)")\n            check_missing_tag_bracket_checkbox.setChecked(qa_settings.get(\'check_missing_tag_bracket\', True))\n            additional_layout.addWidget(check_missing_tag_bracket_checkbox)\n\n            # Silent truncation detection (optional, requires heavy deps)\n            check_truncation_checkbox = self._create_styled_checkbox("Silent truncation detection (compares tail paragraphs via back-translation + embeddings)")\n'), ("                        'check_invalid_tag_mismatch': (check_invalid_tag_mismatch_checkbox, lambda x: x.isChecked()),\n                        'check_invalid_nesting': (check_invalid_nesting_checkbox, lambda x: x.isChecked()),\n                        'check_silent_truncation': (check_truncation_checkbox, lambda x: x.isChecked()),\n                        'truncation_cheap_threshold': (truncation_cheap_slider, lambda x: x.value()),\n", "                        'check_invalid_tag_mismatch': (check_invalid_tag_mismatch_checkbox, lambda x: x.isChecked()),\n                        'check_invalid_nesting': (check_invalid_nesting_checkbox, lambda x: x.isChecked()),\n                        'check_missing_tag_bracket': (check_missing_tag_bracket_checkbox, lambda x: x.isChecked()),\n                        'check_silent_truncation': (check_truncation_checkbox, lambda x: x.isChecked()),\n                        'truncation_cheap_threshold': (truncation_cheap_slider, lambda x: x.value()),\n"), ("                    ('check_invalid_tag_mismatch', check_invalid_tag_mismatch_checkbox, False),\n                    ('check_invalid_nesting', check_invalid_nesting_checkbox, False),\n                    ('check_silent_truncation', check_truncation_checkbox, False),\n                    ('truncation_cheap_threshold', truncation_cheap_slider, 12),\n", "                    ('check_invalid_tag_mismatch', check_invalid_tag_mismatch_checkbox, False),\n                    ('check_invalid_nesting', check_invalid_nesting_checkbox, False),\n                    ('check_missing_tag_bracket', check_missing_tag_bracket_checkbox, True),\n                    ('check_silent_truncation', check_truncation_checkbox, False),\n                    ('truncation_cheap_threshold', truncation_cheap_slider, 12),\n"), ('                    check_invalid_tag_mismatch_checkbox.setChecked(False)\n                    check_invalid_nesting_checkbox.setChecked(False)\n                    check_truncation_checkbox.setChecked(False)\n                    truncation_cheap_slider.setValue(12)\n', '                    check_invalid_tag_mismatch_checkbox.setChecked(False)\n                    check_invalid_nesting_checkbox.setChecked(False)\n                    check_missing_tag_bracket_checkbox.setChecked(True)\n                    check_truncation_checkbox.setChecked(False)\n                    truncation_cheap_slider.setValue(12)\n')], 'src/qa_scan_runtime.py': [('        "check_invalid_tag_mismatch": False,\n        "check_invalid_nesting": False,\n        "check_silent_truncation": False,\n        "truncation_cheap_threshold": 12,\n', '        "check_invalid_tag_mismatch": False,\n        "check_invalid_nesting": False,\n        "check_missing_tag_bracket": True,\n        "check_silent_truncation": False,\n        "truncation_cheap_threshold": 12,\n'), ('        "QA_CHECK_MISSING_BEAUTIFULSOUP_TAGS": "1" if settings.get("check_missing_beautifulsoup_tags", False) else "0",\n        "QA_CHECK_INVALID_NESTING": "1" if settings.get("check_invalid_nesting", False) else "0",\n        "QA_CHECK_SILENT_TRUNCATION": "1" if settings.get("check_silent_truncation", False) else "0",\n        "QA_CHECK_POTENTIAL_TRUNCATION": "1" if settings.get("check_potential_truncation", False) else "0",\n', '        "QA_CHECK_MISSING_BEAUTIFULSOUP_TAGS": "1" if settings.get("check_missing_beautifulsoup_tags", False) else "0",\n        "QA_CHECK_INVALID_NESTING": "1" if settings.get("check_invalid_nesting", False) else "0",\n        "QA_CHECK_MISSING_TAG_BRACKET": "1" if settings.get("check_missing_tag_bracket", True) else "0",\n        "QA_CHECK_SILENT_TRUNCATION": "1" if settings.get("check_silent_truncation", False) else "0",\n        "QA_CHECK_POTENTIAL_TRUNCATION": "1" if settings.get("check_potential_truncation", False) else "0",\n')], 'src/scan_html_folder.py': [('from bs4 import BeautifulSoup, Comment, NavigableString\nfrom bs4.element import CData, RubyTextString, RubyParenthesisString\nfrom _empty_attr_fix import count_empty_attr_tags, find_empty_attr_tags\nfrom langdetect import detect, LangDetectException\nfrom difflib import SequenceMatcher\n', 'from bs4 import BeautifulSoup, Comment, NavigableString\nfrom bs4.element import CData, RubyTextString, RubyParenthesisString\nfrom _empty_attr_fix import (\n    count_empty_attr_tags,\n    count_missing_tag_brackets,\n    find_empty_attr_tags,\n    find_missing_tag_brackets,\n)\nfrom langdetect import detect, LangDetectException\nfrom difflib import SequenceMatcher\n'), ('        # Detect translation artifacts\n        artifacts = []\n        check_translation_artifacts = qa_settings.get(\'check_translation_artifacts\', False)\n        check_ai_artifacts = qa_settings.get(\'check_ai_artifacts\', False)\n        # Allow checking artifacts in quick scan if explicitly enabled (regex is fast)\n        if check_translation_artifacts:\n            artifacts = detect_translation_artifacts(raw_text)\n            \n            # Check for "LLM Token Fix"-style empty-attribute tag hallucinations\n', '        # Detect translation artifacts\n        artifacts = []\n        # Checked in RAW HTML under its own toggle: these tags are not visible text.\n        if raw_file_content and qa_settings.get(\'check_missing_tag_bracket\', True):\n            missing_bracket_count = count_missing_tag_brackets(raw_file_content)\n            if missing_bracket_count:\n                artifacts.append({\n                    \'type\': \'empty_attribute_tags\',\n                    \'count\': missing_bracket_count,\n                    \'examples\': find_missing_tag_brackets(raw_file_content, limit=10),\n                    \'severity\': \'medium\'\n                })\n        check_translation_artifacts = qa_settings.get(\'check_translation_artifacts\', False)\n        check_ai_artifacts = qa_settings.get(\'check_ai_artifacts\', False)\n        # Allow checking artifacts in quick scan if explicitly enabled (regex is fast)\n        if check_translation_artifacts:\n            artifacts.extend(detect_translation_artifacts(raw_text))\n            \n            # Check for "LLM Token Fix"-style empty-attribute tag hallucinations\n'), ("            'sdlxliff_min_source_paragraph_tags': 20,\n            'check_invalid_nesting': False,\n            'check_silent_truncation': False,\n            'check_potential_truncation': False,\n", "            'sdlxliff_min_source_paragraph_tags': 20,\n            'check_invalid_nesting': False,\n            'check_missing_tag_bracket': True,\n            'check_silent_truncation': False,\n            'check_potential_truncation': False,\n"), ('        )\n    log(f"   ✓ Invalid nesting check: {\'ENABLED\' if qa_settings.get(\'check_invalid_nesting\', False) else \'DISABLED\'}") \n    log(f"   ✓ Silent truncation check: {\'ENABLED\' if qa_settings.get(\'check_silent_truncation\', False) else \'DISABLED\'}")\n    log(f"   ✓ Potential truncation check: {\'ENABLED\' if qa_settings.get(\'check_potential_truncation\', False) else \'DISABLED\'}")\n', '        )\n    log(f"   ✓ Invalid nesting check: {\'ENABLED\' if qa_settings.get(\'check_invalid_nesting\', False) else \'DISABLED\'}") \n    log(f"   ✓ Missing tag bracket check: {\'ENABLED\' if qa_settings.get(\'check_missing_tag_bracket\', True) else \'DISABLED\'}")\n    log(f"   ✓ Silent truncation check: {\'ENABLED\' if qa_settings.get(\'check_silent_truncation\', False) else \'DISABLED\'}")\n    log(f"   ✓ Potential truncation check: {\'ENABLED\' if qa_settings.get(\'check_potential_truncation\', False) else \'DISABLED\'}")\n')]}


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
    {
        "date": "2026-10-23",
        "what": "New 'Payload cleanup limit (MB)' setting (Other Settings, above the custom OpenAI endpoint; "
                "default 400, 0 = never clean up) for the startup / exit trim of Payloads and http_requests",
        "source": {
            "src/settings_persistence.py": [_PAYLOAD_CLEANUP_SETTINGS_ROW],
            "src/owner_state.py": [_PAYLOAD_CLEANUP_INIT],
            "src/translator_gui.py": [_PAYLOAD_CLEANUP_SETTINGS_ROW_GUI],
        },
        "values": {},
    },
    {
        "date": "2026-10-23",
        "what": "New Payloads > 'Auto-update' setting (payload_summary_auto_update, off): rebuild the readable "
                "payload summary after a translation / glossary run is stopped",
        "source": {
            "src/settings_schema.py": _PAYLOAD_SUMMARY_AUTO_SCHEMA,
            "src/settings_schema_data.py": _PAYLOAD_SUMMARY_AUTO_SCHEMA_DATA,
        },
        "values": {},
    },
    {
        "date": "2026-10-23",
        "what": "Payloads > 'Auto-update' also rebuilds the payload summary after 'Translation completed "
                "successfully!' and after the glossary extraction summary; new Payloads > 'Decode images' "
                "setting (payload_summary_decode_images, off)",
        "source": {
            "src/settings_schema.py": _PAYLOAD_SUMMARY_FINISH_SCHEMA,
            "src/settings_schema_data.py": _PAYLOAD_SUMMARY_FINISH_SCHEMA_DATA,
            "src/text_jobs.py": _PAYLOAD_SUMMARY_FINISH_TEXT_JOBS,
            "src/translation_pipeline.py": _PAYLOAD_SUMMARY_FINISH_PIPELINE,
            # the pinned older sources the guard tests start from
            "src/translator_gui.py": _PAYLOAD_SUMMARY_LEGACY_TRANSLATOR_GUI,
            "src/owner_state.py": _PAYLOAD_SUMMARY_LEGACY_OWNER_STATE,
            "src/settings_persistence.py": _PAYLOAD_SUMMARY_LEGACY_SETTINGS_PERSISTENCE,
        },
        "values": {},
    },
    {
        "date": "2026-10-24",
        "what": "The glossary name check minimum length / Auto-fix strength and the payload cleanup limit "
                "in the pinned translator_gui (its _init_variables and the startup / exit cache sweep)",
        "source": {
            "src/translator_gui.py": _OLD_SETTINGS_INIT_GUI + _PAYLOAD_CLEANUP_SWEEP_GUI,
        },
        "values": {},
    },
    {
        "date": "2026-10-24",
        "what": "Arena's reCAPTCHA rejection (autharena_verification_error) is logged without the full "
                "traceback, like config / entitlement errors",
        "source": {
            "src/text_jobs.py": _ARENA_VERIFICATION_NO_TRACEBACK,
        },
        "values": {},
    },
    {
        "date": "2026-10-27",
        "what": "New glossary name check setting 'Names keep their numbers' (glossary_name_check_match_numbers, "
                "on): a candidate with another number is never a respelling, and a name whose number is glued "
                "to the word before it counts as found",
        "source": {
            **{rel: [pair for pair in pairs] for rel, pairs in _MATCH_NUMBERS_LEGACY.items()},
            **_MATCH_NUMBERS_SCHEMA,
        },
        "values": {},
    },
    {
        "date": "2026-10-27",
        "what": "New glossary name check setting 'Chapters checked at once' (glossary_name_check_workers, "
                "0 = Auto: the count Maximum parallel workers gave before)",
        "source": _NAME_CHECK_WORKERS_LEGACY,
        "values": {},
    },
    {
        "date": "2026-10-28",
        "what": "New glossary name check settings 'NER plural exclusion' (on) and 'NER tense exclusion (not "
                "recommended)' (off), English output only, and the entry types both skip ('Skip plural / "
                "tense exclusion for', default character and nicknames); 'Chapters checked at once' up to "
                "the cores (cap 32)",
        "source": _NAME_CHECK_INFLECTION_LEGACY,
        "values": {},
    },
    {
        "date": "2026-10-28",
        "what": "QA scan: the batches a dead process-pool worker leaves (BrokenProcessPool, one traceback per "
                "batch in v9.15.6) run again on threads; a worker that dies in native code writes its "
                "traceback to the app's crash.log",
        "source": {
            "src/scan_html_folder.py": _QA_BROKEN_POOL_SCANNER,
        },
        "values": {},
    },
    {
        "date": "2026-10-28",
        "what": "New setting 'Block other scripts in translations (adaptive, all models)' (block_other_scripts, "
                "off; Anti-Duplicate Parameters > Logit Bias): the on-device adaptive script ban for every API "
                "model (OpenAI logit bias where the tokenizer is known, an output guard with a retry elsewhere)",
        "source": {
            **_BLOCK_OTHER_SCRIPTS_LEGACY,
            **_BLOCK_OTHER_SCRIPTS_SCHEMA,
        },
        "values": {},
    },
    {
        "date": "2026-10-28",
        "what": "New setting 'Allow emoticons and symbols from other scripts (kaomoji)' "
                "(script_ban_allow_kaomoji, off; Anti-Duplicate Parameters > Logit Bias, and under the "
                "on-device switch on mobile): the script ban lets kaomoji punctuation and the source's "
                "emoticon characters through",
        "source": {
            **_SCRIPT_BAN_ALLOW_KAOMOJI_LEGACY,
            **_SCRIPT_BAN_ALLOW_KAOMOJI_SCHEMA,
        },
        "values": {},
    },
    {
        "date": "2026-10-29",
        "what": "config.json is never left truncated: every writer goes through the atomic write "
                "(no direct-write fallback), here the manga settings dialog's save fallbacks and the "
                "QA scanner's custom-mode save",
        "source": {
            "src/manga_settings_dialog.py": _ATOMIC_CONFIG_MANGA_DIALOG,
            "src/QA_Scanner_GUI.py": _ATOMIC_CONFIG_QA_SCANNER,
        },
        "values": {},
    },
    {
        "date": "2026-10-30",
        "what": "Features a build leaves out say 'Not supported in this build' first (owner): the QA "
                "scanner's greyed silent truncation line (the NoCuda build no longer bundles its deps)",
        "source": {
            "src/QA_Scanner_GUI.py": [
                ('QLabel(f"⚠️ Missing dependencies: {missing_str}\\n"',
                 'QLabel(f"⚠️ Not supported in this build. Missing dependencies: {missing_str}\\n"'),
            ],
        },
        "values": {},
    },
    {
        "date": "2026-10-30",
        "what": "New setting 'Whitelist source text' (script_ban_whitelist_source, off; Anti-Duplicate "
                "Parameters > Logit Bias, Ollama Settings, and under the on-device switch on mobile): every "
                "banned request may use the characters of its own source text",
        "source": {
            **_SCRIPT_BAN_WHITELIST_LEGACY,
            **_SCRIPT_BAN_WHITELIST_SCHEMA,
        },
        "values": {},
    },
    {
        "date": "2026-10-30",
        "what": "New settings 'Forced Truncation' (forced_truncation_enabled, off; forced_truncation_seconds, 3; "
                "Anti-Duplicate Parameters > Logit Bias): an answer repeating itself for that long is stopped "
                "with finish reason length unless the source repeats as often",
        "source": {
            **_FORCED_TRUNCATION_LEGACY,
            **_FORCED_TRUNCATION_SCHEMA,
        },
        "values": {},
    },
    {
        "date": "2026-10-30",
        "what": "New setting 'Mark glossary forced truncation as complete' (mark_glossary_forced_truncation_complete, off): a glossary "
                "request stopped by Forced Truncation ends with finish reason stop",
        "source": {
            **_MARK_GLOSSARY_COMPLETE_LEGACY,
            **_MARK_GLOSSARY_COMPLETE_SCHEMA,
        },
        "values": {},
    },
    {
        "date": "2026-11-01",
        "what": "The QA scanner stops at once when a stop flag is raised: a graceful stop too, unless API calls are "
                "in flight (the AI truncation checker), which finish, and a forced stop aborts them",
        "source": {
            "src/scan_html_folder.py": _QA_STOP_SCANNER,
        },
        "values": {},
    },
    {
        "date": "2026-11-01",
        "what": "New setting 'Fix Missing Tag Bracket' (fix_missing_tag_bracket, off) restores the > an LLM dropped "
                "in <p\"Hello\"</p> before parsing; new QA check (check_missing_tag_bracket, on) reports it as an "
                "LLM token issue, repaired by Resolve QA",
        "source": {
            **_MISSING_BRACKET_LEGACY,
            **_MISSING_BRACKET_SCHEMA,
            **_MISSING_BRACKET_QA,
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
