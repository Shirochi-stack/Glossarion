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
