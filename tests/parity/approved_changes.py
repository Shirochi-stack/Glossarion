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



# 2026-11-06 Manga editor Format toolbar (per-box text style): installed on the image preview right after
# its workflow signals are connected (manga_format_toolbar.install_format_toolbar).
_FORMAT_TOOLBAR_INSTALL = (
    "        self.image_preview_widget.translate_all_clicked.connect(lambda: ImageRenderer._on_translate_all_clicked(self))\n"
    "        \n"
    "        # Connect viewer click signal to sync file list selection with current image\n",
    "        self.image_preview_widget.translate_all_clicked.connect(lambda: ImageRenderer._on_translate_all_clicked(self))\n"
    "        \n"
    "        # Format toolbar (per-box font, size, bold / italic / underline, alignment, colour)\n"
    "        try:\n"
    "            from manga_format_toolbar import install_format_toolbar\n"
    "            install_format_toolbar(self.image_preview_widget, self)\n"
    "        except Exception as format_err:\n"
    "            print(f\"[FORMAT] Format toolbar unavailable: {format_err}\")\n"
    "        \n"
    "        # Connect viewer click signal to sync file list selection with current image\n",
)

# 2026-11-06 A box's right-click menu gets a "Text format" submenu: the Format toolbar's changes
# (bold / italic / underline, alignment, size, font, colour, reset) from the box itself.
_FORMAT_CONTEXT_MENU = (
    '                    menu.addAction(clean_action)\n'
    '                    \n'
    '                    # Add "Delete Selected" option\n',
    '                    menu.addAction(clean_action)\n'
    '                    \n'
    "                    # Text format submenu (the preview's Format toolbar: per-box font, size, style, colour)\n"
    '                    try:\n'
    "                        format_toolbar = getattr(getattr(self, 'image_preview_widget', None), 'format_toolbar', None)\n"
    '                        if format_toolbar is not None:\n'
    '                            format_toolbar.add_context_menu(menu, actual_index)\n'
    '                    except Exception as format_err:\n'
    '                        print(f"[FORMAT] Text format menu unavailable: {format_err}")\n'
    '                    \n'
    '                    # Add "Delete Selected" option\n',
)

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

# 2026-11-01 "Fix Missing Tag Bracket" (fix_missing_tag_bracket / FIX_MISSING_TAG_BRACKET, on) and its QA check
# (check_missing_tag_bracket / QA_CHECK_MISSING_TAG_BRACKET, on)
_MISSING_BRACKET_LEGACY = {'src/translator_gui.py': [("                ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n                ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n", "                ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n                ('fix_missing_tag_bracket', ['fix_missing_tag_bracket_var'], True, bool),\n                ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n"), ("            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n", "            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('fix_missing_tag_bracket', ['fix_missing_tag_bracket_var'], True, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n"), ("            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            # Keep existing variables intact\n", "            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            ('fix_missing_tag_bracket_var', 'fix_missing_tag_bracket', True),\n            # Keep existing variables intact\n"), ("            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n", "            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            'FIX_MISSING_TAG_BRACKET': '1' if getattr(self, 'fix_missing_tag_bracket_var', True) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n"), ("                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n\n                # Post-translation scanning phase\n", "                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n                ('FIX_MISSING_TAG_BRACKET', '1' if getattr(self, 'fix_missing_tag_bracket_var', True) else '0'),\n\n                # Post-translation scanning phase\n"), ("        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n", "        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_missing_tag_bracket_var = self.config.get('fix_missing_tag_bracket', True)\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n"), ("        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n", "        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_MISSING_TAG_BRACKET'] = '1' if self.fix_missing_tag_bracket_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n")], 'src/owner_state.py': [("            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            # Keep existing variables intact\n", "            ('remove_excessive_empty_tags_all_models_var', 'remove_excessive_empty_tags_all_models', False),\n            ('fix_missing_tag_bracket_var', 'fix_missing_tag_bracket', True),\n            # Keep existing variables intact\n"), ("        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n", "        self.fix_empty_attr_tags_bs_var = self.fix_empty_attr_tags_extract_var\n        self.config['fix_empty_attr_tags_extract'] = self.fix_empty_attr_tags_extract_var\n        self.fix_missing_tag_bracket_var = self.config.get('fix_missing_tag_bracket', True)\n        self.fix_stray_p_gt_epub_var = self.config.get('fix_stray_p_gt_epub', False)\n        self.fix_stray_p_gt_bs_var = self.config.get('fix_stray_p_gt_bs', False)\n"), ("        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n", "        os.environ['FIX_EMPTY_ATTR_TAGS_EXTRACT'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_EMPTY_ATTR_TAGS_BS'] = '1' if self.fix_empty_attr_tags_extract_var else '0'\n        os.environ['FIX_MISSING_TAG_BRACKET'] = '1' if self.fix_missing_tag_bracket_var else '0'\n        os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.fix_stray_p_gt_epub_var else '0'\n        os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.fix_stray_p_gt_bs_var else '0'\n")], 'src/settings_persistence.py': [("            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n", "            ('remove_excessive_empty_tags_all_models', ['remove_excessive_empty_tags_all_models_var'], False, bool),\n            ('fix_missing_tag_bracket', ['fix_missing_tag_bracket_var'], True, bool),\n            ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),\n")], 'src/run_env.py': [("            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n", "            'REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE': '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0',\n            'REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS': '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0',\n            'FIX_MISSING_TAG_BRACKET': '1' if getattr(self, 'fix_missing_tag_bracket_var', True) else '0',\n            # Scanning/QA settings\n            'SCAN_PHASE_ENABLED': '1' if getattr(self, 'scan_phase_enabled_var', self.config.get('scan_phase_enabled', False)) else '0',\n"), ("                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n\n                # Post-translation scanning phase\n", "                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ON_DEVICE', '1' if getattr(self, 'remove_excessive_empty_tags_on_device_var', True) else '0'),\n                ('REMOVE_EXCESSIVE_EMPTY_TAGS_ALL_MODELS', '1' if getattr(self, 'remove_excessive_empty_tags_all_models_var', False) else '0'),\n                ('FIX_MISSING_TAG_BRACKET', '1' if getattr(self, 'fix_missing_tag_bracket_var', True) else '0'),\n\n                # Post-translation scanning phase\n")]}
_MISSING_BRACKET_SCHEMA = {'src/settings_schema_data.py': [(" 'remove_excessive_empty_tags_on_device',\n 'remove_excessive_empty_tags_all_models',\n 'emergency_paragraph_restore',\n 'emergency_image_restore',\n", " 'remove_excessive_empty_tags_on_device',\n 'remove_excessive_empty_tags_all_models',\n 'fix_missing_tag_bracket',\n 'emergency_paragraph_restore',\n 'emergency_image_restore',\n"), ("  ('value', False),\n  ('bool',)),\n ('emergency_paragraph_restore', ('emergency_restore_var',), ('value', False), ('bool',)),\n ('emergency_image_restore', ('emergency_image_restore_var',), ('value', False), ('bool',)),\n", "  ('value', False),\n  ('bool',)),\n ('fix_missing_tag_bracket', ('fix_missing_tag_bracket_var',), ('value', True), ('bool',)),\n ('emergency_paragraph_restore', ('emergency_restore_var',), ('value', False), ('bool',)),\n ('emergency_image_restore', ('emergency_image_restore_var',), ('value', False), ('bool',)),\n"), ("  'remove_excessive_empty_tags_all_models',\n  ('value', False)),\n ('enable_image_translation_var', 'enable_image_translation', ('value', False)),\n ('process_webnovel_images_var', 'process_webnovel_images', ('value', True)),\n", "  'remove_excessive_empty_tags_all_models',\n  ('value', False)),\n ('fix_missing_tag_bracket_var', 'fix_missing_tag_bracket', ('value', True)),\n ('enable_image_translation_var', 'enable_image_translation', ('value', False)),\n ('process_webnovel_images_var', 'process_webnovel_images', ('value', True)),\n"), ("     'ui_group': 'Processing Options',\n     'ui_sites': ('other.processing',)},\n    'fix_stray_p_gt_bs': {'converter': ('bool',),\n     'default': False,\n", "     'ui_group': 'Processing Options',\n     'ui_sites': ('other.processing',)},\n    'fix_missing_tag_bracket': {'converter': ('bool',),\n     'default': True,\n     'default_source': 'init',\n     'dialog_default': True,\n     'env': (('FIX_MISSING_TAG_BRACKET', 'translation', True),\n             ('FIX_MISSING_TAG_BRACKET', 'startup', '<none>'),\n             ('FIX_MISSING_TAG_BRACKET', 'startup', True)),\n     'init_default': True,\n     'label': 'Fix Missing Tag Bracket - LLM Token Fix',\n     'origins': ('settings_map', 'init', 'dialog', 'env'),\n     'save_default': True,\n     'type': 'bool',\n     'ui_group': 'Processing Options',\n     'ui_sites': ('other.processing',),\n     'var_names': ('fix_missing_tag_bracket_var',)},\n    'fix_stray_p_gt_bs': {'converter': ('bool',),\n     'default': False,\n"), ("     'ui_group': 'Word Count Analysis',\n     'ui_sites': ('qa',)},\n    'qa_scanner_settings.check_multiple_headers': {'default': True,\n     'default_source': 'nested',\n", '     \'ui_group\': \'Word Count Analysis\',\n     \'ui_sites\': (\'qa\',)},\n    \'qa_scanner_settings.check_missing_tag_bracket\': {\'default\': True,\n     \'default_source\': \'nested\',\n     \'dialog_default\': True,\n     \'env\': ((\'QA_CHECK_MISSING_TAG_BRACKET\', \'qa\', True),),\n     \'label\': \'Check for missing tag brackets (<p"Hello"</p>)\',\n     \'nested_default\': True,\n     \'origins\': (\'nested\', \'dialog\', \'env\'),\n     \'parent\': \'qa_scanner_settings\',\n     \'type\': \'bool\',\n     \'ui_group\': \'Additional Checks\',\n     \'ui_sites\': (\'qa\',)},\n    \'qa_scanner_settings.check_multiple_headers\': {\'default\': True,\n     \'default_source\': \'nested\',\n')]}
_MISSING_BRACKET_QA = {'src/QA_Scanner_GUI.py': [("                    'sdlxliff_min_source_paragraph_tags': 20,\n                    'check_invalid_nesting': False,\n                    'cache_enabled': True,\n                    'cache_auto_size': False,\n", "                    'sdlxliff_min_source_paragraph_tags': 20,\n                    'check_invalid_nesting': False,\n                    'check_missing_tag_bracket': True,\n                    'cache_enabled': True,\n                    'cache_auto_size': False,\n"), ('            additional_layout.addWidget(check_invalid_nesting_checkbox)\n\n            # Silent truncation detection (optional, requires heavy deps)\n            check_truncation_checkbox = self._create_styled_checkbox("Silent truncation detection (compares tail paragraphs via back-translation + embeddings)")\n', '            additional_layout.addWidget(check_invalid_nesting_checkbox)\n\n            # Missing tag bracket check (LLM token issue, separate toggle)\n            check_missing_tag_bracket_checkbox = self._create_styled_checkbox("Check for missing tag brackets (<p\\"Hello\\"</p>)")\n            check_missing_tag_bracket_checkbox.setChecked(qa_settings.get(\'check_missing_tag_bracket\', True))\n            additional_layout.addWidget(check_missing_tag_bracket_checkbox)\n\n            # Silent truncation detection (optional, requires heavy deps)\n            check_truncation_checkbox = self._create_styled_checkbox("Silent truncation detection (compares tail paragraphs via back-translation + embeddings)")\n'), ("                        'check_invalid_tag_mismatch': (check_invalid_tag_mismatch_checkbox, lambda x: x.isChecked()),\n                        'check_invalid_nesting': (check_invalid_nesting_checkbox, lambda x: x.isChecked()),\n                        'check_silent_truncation': (check_truncation_checkbox, lambda x: x.isChecked()),\n                        'truncation_cheap_threshold': (truncation_cheap_slider, lambda x: x.value()),\n", "                        'check_invalid_tag_mismatch': (check_invalid_tag_mismatch_checkbox, lambda x: x.isChecked()),\n                        'check_invalid_nesting': (check_invalid_nesting_checkbox, lambda x: x.isChecked()),\n                        'check_missing_tag_bracket': (check_missing_tag_bracket_checkbox, lambda x: x.isChecked()),\n                        'check_silent_truncation': (check_truncation_checkbox, lambda x: x.isChecked()),\n                        'truncation_cheap_threshold': (truncation_cheap_slider, lambda x: x.value()),\n"), ("                    ('check_invalid_tag_mismatch', check_invalid_tag_mismatch_checkbox, False),\n                    ('check_invalid_nesting', check_invalid_nesting_checkbox, False),\n                    ('check_silent_truncation', check_truncation_checkbox, False),\n                    ('truncation_cheap_threshold', truncation_cheap_slider, 12),\n", "                    ('check_invalid_tag_mismatch', check_invalid_tag_mismatch_checkbox, False),\n                    ('check_invalid_nesting', check_invalid_nesting_checkbox, False),\n                    ('check_missing_tag_bracket', check_missing_tag_bracket_checkbox, True),\n                    ('check_silent_truncation', check_truncation_checkbox, False),\n                    ('truncation_cheap_threshold', truncation_cheap_slider, 12),\n"), ('                    check_invalid_tag_mismatch_checkbox.setChecked(False)\n                    check_invalid_nesting_checkbox.setChecked(False)\n                    check_truncation_checkbox.setChecked(False)\n                    truncation_cheap_slider.setValue(12)\n', '                    check_invalid_tag_mismatch_checkbox.setChecked(False)\n                    check_invalid_nesting_checkbox.setChecked(False)\n                    check_missing_tag_bracket_checkbox.setChecked(True)\n                    check_truncation_checkbox.setChecked(False)\n                    truncation_cheap_slider.setValue(12)\n')], 'src/qa_scan_runtime.py': [('        "check_invalid_tag_mismatch": False,\n        "check_invalid_nesting": False,\n        "check_silent_truncation": False,\n        "truncation_cheap_threshold": 12,\n', '        "check_invalid_tag_mismatch": False,\n        "check_invalid_nesting": False,\n        "check_missing_tag_bracket": True,\n        "check_silent_truncation": False,\n        "truncation_cheap_threshold": 12,\n'), ('        "QA_CHECK_MISSING_BEAUTIFULSOUP_TAGS": "1" if settings.get("check_missing_beautifulsoup_tags", False) else "0",\n        "QA_CHECK_INVALID_NESTING": "1" if settings.get("check_invalid_nesting", False) else "0",\n        "QA_CHECK_SILENT_TRUNCATION": "1" if settings.get("check_silent_truncation", False) else "0",\n        "QA_CHECK_POTENTIAL_TRUNCATION": "1" if settings.get("check_potential_truncation", False) else "0",\n', '        "QA_CHECK_MISSING_BEAUTIFULSOUP_TAGS": "1" if settings.get("check_missing_beautifulsoup_tags", False) else "0",\n        "QA_CHECK_INVALID_NESTING": "1" if settings.get("check_invalid_nesting", False) else "0",\n        "QA_CHECK_MISSING_TAG_BRACKET": "1" if settings.get("check_missing_tag_bracket", True) else "0",\n        "QA_CHECK_SILENT_TRUNCATION": "1" if settings.get("check_silent_truncation", False) else "0",\n        "QA_CHECK_POTENTIAL_TRUNCATION": "1" if settings.get("check_potential_truncation", False) else "0",\n')], 'src/scan_html_folder.py': [('from bs4 import BeautifulSoup, Comment, NavigableString\nfrom bs4.element import CData, RubyTextString, RubyParenthesisString\nfrom _empty_attr_fix import count_empty_attr_tags, find_empty_attr_tags\nfrom langdetect import detect, LangDetectException\nfrom difflib import SequenceMatcher\n', 'from bs4 import BeautifulSoup, Comment, NavigableString\nfrom bs4.element import CData, RubyTextString, RubyParenthesisString\nfrom _empty_attr_fix import (\n    count_empty_attr_tags,\n    count_missing_tag_brackets,\n    find_empty_attr_tags,\n    find_missing_tag_brackets,\n)\nfrom langdetect import detect, LangDetectException\nfrom difflib import SequenceMatcher\n'), ('        # Detect translation artifacts\n        artifacts = []\n        check_translation_artifacts = qa_settings.get(\'check_translation_artifacts\', False)\n        check_ai_artifacts = qa_settings.get(\'check_ai_artifacts\', False)\n        # Allow checking artifacts in quick scan if explicitly enabled (regex is fast)\n        if check_translation_artifacts:\n            artifacts = detect_translation_artifacts(raw_text)\n            \n            # Check for "LLM Token Fix"-style empty-attribute tag hallucinations\n', '        # Detect translation artifacts\n        artifacts = []\n        # Checked in RAW HTML under its own toggle: these tags are not visible text.\n        if raw_file_content and qa_settings.get(\'check_missing_tag_bracket\', True):\n            missing_bracket_count = count_missing_tag_brackets(raw_file_content)\n            if missing_bracket_count:\n                artifacts.append({\n                    \'type\': \'empty_attribute_tags\',\n                    \'count\': missing_bracket_count,\n                    \'examples\': find_missing_tag_brackets(raw_file_content, limit=10),\n                    \'severity\': \'medium\'\n                })\n        check_translation_artifacts = qa_settings.get(\'check_translation_artifacts\', False)\n        check_ai_artifacts = qa_settings.get(\'check_ai_artifacts\', False)\n        # Allow checking artifacts in quick scan if explicitly enabled (regex is fast)\n        if check_translation_artifacts:\n            artifacts.extend(detect_translation_artifacts(raw_text))\n            \n            # Check for "LLM Token Fix"-style empty-attribute tag hallucinations\n'), ("            'sdlxliff_min_source_paragraph_tags': 20,\n            'check_invalid_nesting': False,\n            'check_silent_truncation': False,\n            'check_potential_truncation': False,\n", "            'sdlxliff_min_source_paragraph_tags': 20,\n            'check_invalid_nesting': False,\n            'check_missing_tag_bracket': True,\n            'check_silent_truncation': False,\n            'check_potential_truncation': False,\n"), ('        )\n    log(f"   ✓ Invalid nesting check: {\'ENABLED\' if qa_settings.get(\'check_invalid_nesting\', False) else \'DISABLED\'}") \n    log(f"   ✓ Silent truncation check: {\'ENABLED\' if qa_settings.get(\'check_silent_truncation\', False) else \'DISABLED\'}")\n    log(f"   ✓ Potential truncation check: {\'ENABLED\' if qa_settings.get(\'check_potential_truncation\', False) else \'DISABLED\'}")\n', '        )\n    log(f"   ✓ Invalid nesting check: {\'ENABLED\' if qa_settings.get(\'check_invalid_nesting\', False) else \'DISABLED\'}") \n    log(f"   ✓ Missing tag bracket check: {\'ENABLED\' if qa_settings.get(\'check_missing_tag_bracket\', True) else \'DISABLED\'}")\n    log(f"   ✓ Silent truncation check: {\'ENABLED\' if qa_settings.get(\'check_silent_truncation\', False) else \'DISABLED\'}")\n    log(f"   ✓ Potential truncation check: {\'ENABLED\' if qa_settings.get(\'check_potential_truncation\', False) else \'DISABLED\'}")\n')]}


# 2026-11-06 Local Model switch: Skip Inpainter's pool lock is refreshed at once (both exits of
# _on_local_model_change; tests/test_manga_inpaint_wait.py)
_SKIP_LOCK_SWITCH_IMAGE_EDIT = (
    "            self._apply_custom_image_edit_ui_state()\n"
    "            self._save_rendering_settings()\n"
    "            return\n"
    "        \n"
    "        # Check for saved path for this model type\n",
    "            self._apply_custom_image_edit_ui_state()\n"
    "            self._sync_skip_inpainter_pool_lock()\n"
    "            self._save_rendering_settings()\n"
    "            return\n"
    "        \n"
    "        # Check for saved path for this model type\n",
)
_SKIP_LOCK_SWITCH_LOCAL = (
    "            self.local_model_status_label.setStyleSheet(\"color: gray;\")\n"
    "\n"
    "        self._apply_custom_image_edit_ui_state()\n"
    "        \n"
    "        self._save_rendering_settings()\n",
    "            self.local_model_status_label.setStyleSheet(\"color: gray;\")\n"
    "\n"
    "        self._apply_custom_image_edit_ui_state()\n"
    "        # Skip Inpainter follows the pool at once, not on the tracker's next 2 s tick\n"
    "        self._sync_skip_inpainter_pool_lock()\n"
    "        \n"
    "        self._save_rendering_settings()\n",
)


# 2026-11-06 Owner: the manga OCR prompt's empty-answer rule also covers illegible text ("no visible / illegible
# text or symbols"), and saved prompts with the previous default lines are migrated to the new ones.
_OCR_PROMPT_ILLEGIBLE_DEFAULT = (
    '    def _default_manga_ocr_prompt(self):\n'
    '        return (\n'
    '            "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\\n\\n"\n'
    '            "The image was selected by an automatic text detector. Detection can be wrong, so verify the image itself.\\n\\n"\n'
    '            "ABSOLUTE RULES:\\n"\n'
    '            "1. OUTPUT ONLY THE VISIBLE TEXT/SYMBOLS - NOTHING ELSE\\n"\n'
    '            "2. NEVER TRANSLATE OR MODIFY\\n"\n'
    '            "3. NEVER EXPLAIN, DESCRIBE, OR COMMENT\\n"\n'
    '            "4. NEVER SAY \\"I can\'t\\" or \\"I cannot\\" or \\"no text\\" or \\"blank image\\"\\n"\n'
    '            "5. IF YOU SEE DOTS, OUTPUT THE DOTS: .\\n"\n'
    '            "6. IF YOU SEE PUNCTUATION, OUTPUT THE PUNCTUATION\\n"\n'
    '            "7. IF YOU SEE A SINGLE CHARACTER, OUTPUT THAT CHARACTER\\n"\n'
    '            "8. IF THERE ARE NO VISIBLE CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]\\n\\n"\n'
    '            "LANGUAGE PRESERVATION:\\n"\n'
    '            "- Korean text → Output in Korean\\n"\n'
    '            "- Japanese text → Output in Japanese\\n"\n'
    '            "- Chinese text → Output in Chinese\\n"\n'
    '            "- English text → Output in English\\n"\n'
    '            "- CJK quotation marks (「」『』【】《》〈〉) → Preserve exactly as shown\\n\\n"\n'
    '            "FORMATTING:\\n"\n'
    '            "- OUTPUT ALL TEXT ON A SINGLE LINE WITH NO LINE BREAKS\\n"\n'
    '            "- NEVER use \\\\n or line breaks in your output\\n\\n"\n'
    '            "FORBIDDEN RESPONSES:\\n"\n'
    '            "- \\"I can see this appears to be...\\"\\n"\n'
    '            "- \\"I cannot make out any clear text...\\"\\n"\n'
    '            "- \\"This appears to be blank...\\"\\n"\n'
    '            "- \\"If there is text present...\\"\\n"\n'
    '            "- ANY explanatory text\\n\\n"\n'
    '            "YOUR ONLY OUTPUT: The exact visible text. Nothing more. Nothing less.\\n"\n'
    '            "If image has a dot → Output: .\\n"\n'
    '            "If image has two dots → Output: . .\\n"\n'
    '            "If image has text → Output: [that text]\\n"\n'
    '            "If image has no visible text or symbols → Output: [AI RESPONSE UNAVAILABLE]"\n'
    '        )\n',
    '    def _default_manga_ocr_prompt(self):\n'
    '        return (\n'
    '            "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\\n\\n"\n'
    '            "The image was selected by an automatic text detector. Detection can be wrong, so verify the image itself.\\n\\n"\n'
    '            "ABSOLUTE RULES:\\n"\n'
    '            "1. OUTPUT ONLY THE VISIBLE TEXT/SYMBOLS - NOTHING ELSE\\n"\n'
    '            "2. NEVER TRANSLATE OR MODIFY\\n"\n'
    '            "3. NEVER EXPLAIN, DESCRIBE, OR COMMENT\\n"\n'
    '            "4. NEVER SAY \\"I can\'t\\" or \\"I cannot\\" or \\"no text\\" or \\"blank image\\"\\n"\n'
    '            "5. IF YOU SEE DOTS, OUTPUT THE DOTS: .\\n"\n'
    '            "6. IF YOU SEE PUNCTUATION, OUTPUT THE PUNCTUATION\\n"\n'
    '            "7. IF YOU SEE A SINGLE CHARACTER, OUTPUT THAT CHARACTER\\n"\n'
    '            "8. IF THERE ARE NO VISIBLE / illegible CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]\\n\\n"\n'
    '            "LANGUAGE PRESERVATION:\\n"\n'
    '            "- Korean text → Output in Korean\\n"\n'
    '            "- Japanese text → Output in Japanese\\n"\n'
    '            "- Chinese text → Output in Chinese\\n"\n'
    '            "- English text → Output in English\\n"\n'
    '            "- CJK quotation marks (「」『』【】《》〈〉) → Preserve exactly as shown\\n\\n"\n'
    '            "FORMATTING:\\n"\n'
    '            "- OUTPUT ALL TEXT ON A SINGLE LINE WITH NO LINE BREAKS\\n"\n'
    '            "- NEVER use \\\\n or line breaks in your output\\n\\n"\n'
    '            "FORBIDDEN RESPONSES:\\n"\n'
    '            "- \\"I can see this appears to be...\\"\\n"\n'
    '            "- \\"I cannot make out any clear text...\\"\\n"\n'
    '            "- \\"This appears to be blank...\\"\\n"\n'
    '            "- \\"If there is text present...\\"\\n"\n'
    '            "- ANY explanatory text\\n\\n"\n'
    '            "YOUR ONLY OUTPUT: The exact visible text. Nothing more. Nothing less.\\n"\n'
    '            "If image has a dot → Output: .\\n"\n'
    '            "If image has two dots → Output: . .\\n"\n'
    '            "If image has text → Output: [that text]\\n"\n'
    '            "If image has no visible / illegible text or symbols → Output: [AI RESPONSE UNAVAILABLE]"\n'
    '        )\n',
)

_OCR_PROMPT_ILLEGIBLE_MIGRATE = (
    '    @staticmethod\n'
    '    def _migrate_legacy_manga_ocr_prompt(prompt):\n'
    '        """Update only saved prompts with the old empty-response OCR rule."""\n'
    '        legacy_rule = "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)"\n'
    '        if not isinstance(prompt, str) or legacy_rule not in prompt:\n'
    '            return prompt\n'
    '\n'
    '        prompt = prompt.replace(\n'
    '            legacy_rule,\n'
    '            "8. IF THERE ARE NO VISIBLE CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]",\n'
    '        ).replace(\n'
    '            "If image is truly blank → Output: [empty/no response]",\n'
    '            "If image has no visible text or symbols → Output: [AI RESPONSE UNAVAILABLE]",\n'
    '        )\n'
    '        detector_note = "The image was selected by an automatic text detector. Detection can be wrong, so verify the image itself."\n'
    '        if detector_note not in prompt:\n'
    '            intro = "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\\n\\n"\n'
    '            prompt = prompt.replace(intro, intro + detector_note + "\\n\\n", 1) if intro in prompt else detector_note + "\\n\\n" + prompt\n'
    '        return prompt\n',
    '    @staticmethod\n'
    '    def _migrate_legacy_manga_ocr_prompt(prompt):\n'
    '        """Update only saved prompts with an older wording of the empty-response OCR rule.\n'
    '\n'
    '        2026-11-06 (owner): "no visible text" became "no visible / illegible text" in rule 8 and the last\n'
    '        line; saved prompts with the previous default lines get the new lines."""\n'
    '        if isinstance(prompt, str):\n'
    '            prompt = prompt.replace(\n'
    '                "8. IF THERE ARE NO VISIBLE CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]",\n'
    '                "8. IF THERE ARE NO VISIBLE / illegible CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]",\n'
    '            ).replace(\n'
    '                "If image has no visible text or symbols → Output: [AI RESPONSE UNAVAILABLE]",\n'
    '                "If image has no visible / illegible text or symbols → Output: [AI RESPONSE UNAVAILABLE]",\n'
    '            )\n'
    '        legacy_rule = "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)"\n'
    '        if not isinstance(prompt, str) or legacy_rule not in prompt:\n'
    '            return prompt\n'
    '\n'
    '        prompt = prompt.replace(\n'
    '            legacy_rule,\n'
    '            "8. IF THERE ARE NO VISIBLE / illegible CHARACTERS, PUNCTUATION, OR SYMBOLS TO TRANSCRIBE, OUTPUT EXACTLY [AI RESPONSE UNAVAILABLE]",\n'
    '        ).replace(\n'
    '            "If image is truly blank → Output: [empty/no response]",\n'
    '            "If image has no visible / illegible text or symbols → Output: [AI RESPONSE UNAVAILABLE]",\n'
    '        )\n'
    '        detector_note = "The image was selected by an automatic text detector. Detection can be wrong, so verify the image itself."\n'
    '        if detector_note not in prompt:\n'
    '            intro = "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\\n\\n"\n'
    '            prompt = prompt.replace(intro, intro + detector_note + "\\n\\n", 1) if intro in prompt else detector_note + "\\n\\n" + prompt\n'
    '        return prompt\n',
)


# 2026-11-06 Manga Glossary Workflow: "Include page images in glossary generation" (manga_glossary_include_images,
# default on): the glossary requests carry the page images (tall pages chunked like translation) in page batches
_MANGA_GLOSSARY_PAGE_IMAGES = [
    (
        "        self.manga_glossary_enabled_value = config.get('manga_glossary_enabled', False)\n"
        "        self.manga_glossary_prompt = config.get('manga_glossary_prompt', self._default_manga_glossary_prompt())\n"
        ,
        "        self.manga_glossary_enabled_value = config.get('manga_glossary_enabled', False)\n"
        "        self.manga_glossary_include_images_value = config.get('manga_glossary_include_images', True)\n"
        "        self.manga_glossary_prompt = config.get('manga_glossary_prompt', self._default_manga_glossary_prompt())\n"
    ),
    (
        '                self.manga_glossary_enabled_value = bool(self.manga_glossary_checkbox.isChecked())\n'
        "            if hasattr(self, 'manga_compress_glossary_checkbox'):\n"
        ,
        '                self.manga_glossary_enabled_value = bool(self.manga_glossary_checkbox.isChecked())\n'
        "            if hasattr(self, 'manga_glossary_include_images_checkbox'):\n"
        '                self.manga_glossary_include_images_value = bool(self.manga_glossary_include_images_checkbox.isChecked())\n'
        "            if hasattr(self, 'manga_compress_glossary_checkbox'):\n"
    ),
    (
        "                self.main_gui.config['manga_glossary_enabled'] = bool(self.manga_glossary_enabled_value)\n"
        "            if hasattr(self, 'manga_glossary_prompt'):\n"
        ,
        "                self.main_gui.config['manga_glossary_enabled'] = bool(self.manga_glossary_enabled_value)\n"
        "            if hasattr(self, 'manga_glossary_include_images_checkbox'):\n"
        '                self.manga_glossary_include_images_value = bool(self.manga_glossary_include_images_checkbox.isChecked())\n'
        "            if hasattr(self, 'manga_glossary_include_images_value'):\n"
        "                self.main_gui.config['manga_glossary_include_images'] = bool(self.manga_glossary_include_images_value)\n"
        "            if hasattr(self, 'manga_glossary_prompt'):\n"
    ),
    (
        '        valid_entries_before_dedupe: List[Dict[str, Any]],\n'
        '    ) -> None:\n'
        '        """Save optional manga glossary generation debug artifacts."""\n'
        '        if not self._manga_glossary_debug_ocr_enabled():\n'
        '            return\n'
        ,
        '        valid_entries_before_dedupe: List[Dict[str, Any]],\n'
        '        image_batches: Optional[List[Dict[str, Any]]] = None,\n'
        '    ) -> None:\n'
        '        """Save optional manga glossary generation debug artifacts.\n'
        '\n'
        '        ``image_batches`` (the page-image requests, when page images were sent) adds\n'
        '        image_batches.json: per request the pages and images (whole page / chunk) it carried."""\n'
        '        if not self._manga_glossary_debug_ocr_enabled():\n'
        '            return\n'
    ),
    (
        '            "note": "Raw/valid entries are saved before manga glossary deduplication.",\n'
        '        }\n'
        '\n'
        '        written = []\n'
        ,
        '            "note": "Raw/valid entries are saved before manga glossary deduplication.",\n'
        '        }\n'
        '        image_batches_payload = None\n'
        '        if image_batches is not None:\n'
        '            image_batches_payload = [\n'
        '                {\n'
        '                    "request": batch.get("request"),\n'
        '                    "pages": batch.get("pages", []),\n'
        '                    "images": batch.get("images", []),\n'
        '                    "response_chars": len(batch.get("response") or ""),\n'
        '                    "parsed_entries": len(batch.get("entries") or []),\n'
        '                }\n'
        '                for batch in image_batches\n'
        '            ]\n'
        '            manifest["image_requests"] = len(image_batches_payload)\n'
        '            manifest["images_sent"] = sum(len(batch["images"]) for batch in image_batches_payload)\n'
        '\n'
        '        written = []\n'
    ),
    (
        '                    json.dumps(self._manga_glossary_debug_json_safe(valid_entries_before_dedupe or []), ensure_ascii=False, indent=2),\n'
        '                )\n'
        '                self._write_manga_glossary_debug_file(\n'
        ,
        '                    json.dumps(self._manga_glossary_debug_json_safe(valid_entries_before_dedupe or []), ensure_ascii=False, indent=2),\n'
        '                )\n'
        '                if image_batches_payload is not None:\n'
        '                    self._write_manga_glossary_debug_file(\n'
        '                        debug_dir,\n'
        '                        "image_batches.json",\n'
        '                        json.dumps(image_batches_payload, ensure_ascii=False, indent=2),\n'
        '                    )\n'
        '                self._write_manga_glossary_debug_file(\n'
    ),
    (
        "        return bool(getattr(self, 'manga_glossary_debug_ocr_text_value', self.main_gui.config.get('manga_glossary_debug_ocr_text', False)))\n"
        ,
        "        return bool(getattr(self, 'manga_glossary_debug_ocr_text_value', self.main_gui.config.get('manga_glossary_debug_ocr_text', False)))\n"
        '\n'
        '    def _manga_glossary_include_images_enabled(self) -> bool:\n'
        '        """Return True when manga glossary generation should send the page images with the OCR text."""\n'
        '        try:\n'
        "            if hasattr(self, 'manga_glossary_include_images_checkbox'):\n"
        '                return bool(self.manga_glossary_include_images_checkbox.isChecked())\n'
        '        except Exception:\n'
        '            pass\n'
        "        return bool(getattr(self, 'manga_glossary_include_images_value', self.main_gui.config.get('manga_glossary_include_images', True)))\n"
    ),
    (
        "            previous_session_context = getattr(client, 'current_session_context', None)\n"
        '            try:\n'
        '                response, _finish_reason, _raw_obj = send_with_interrupt(\n'
        '                    messages=messages,\n'
        '                    client=client,\n'
        '                    temperature=temperature,\n'
        '                    max_tokens=max_tokens,\n'
        '                    stop_check_fn=lambda: self.stop_flag.is_set(),\n'
        "                    context='glossary',\n"
        '                )\n'
        '            finally:\n'
        ,
        "            previous_session_context = getattr(client, 'current_session_context', None)\n"
        '            image_batches = None\n'
        '            try:\n'
        '                if self._manga_glossary_include_images_enabled():\n'
        '                    image_batches = self._send_manga_glossary_page_image_batches(ocr_pages, client, temperature, max_tokens)\n'
        '                if image_batches is None:\n'
        '                    response, _finish_reason, _raw_obj = send_with_interrupt(\n'
        '                        messages=messages,\n'
        '                        client=client,\n'
        '                        temperature=temperature,\n'
        '                        max_tokens=max_tokens,\n'
        '                        stop_check_fn=lambda: self.stop_flag.is_set(),\n'
        "                        context='glossary',\n"
        '                    )\n'
        '            finally:\n'
    ),
    (
        '                    pass\n'
        '            response_text = self._normalize_manga_api_response_text(response)\n'
        '            self._log(f"📥 Manga glossary response received ({len(response_text)} chars)", "info")\n'
        '\n'
        '            parsed_entries = parse_api_response(response_text)\n'
        '            entries = []\n'
        ,
        '                    pass\n'
        '            if image_batches is None:\n'
        '                response_text = self._normalize_manga_api_response_text(response)\n'
        '                self._log(f"📥 Manga glossary response received ({len(response_text)} chars)", "info")\n'
        '\n'
        '                parsed_entries = parse_api_response(response_text)\n'
        '            else:\n'
        '                # the entries of every page-image request, merged; validated and de-duplicated below\n'
        '                response_text = "\\n\\n".join(\n'
        '                    f"=== REQUEST {batch[\'request\']}/{len(image_batches)} ===\\n{batch[\'response\']}" for batch in image_batches\n'
        '                )\n'
        "                parsed_entries = [entry for batch in image_batches for entry in batch['entries']]\n"
        '            entries = []\n'
    ),
    (
        '                entries,\n'
        '            )\n'
        ,
        '                entries,\n'
        '                image_batches=image_batches,\n'
        '            )\n'
    ),
    (
        '            self._restore_manga_glossary_env(previous_env)\n'
        ,
        '            self._restore_manga_glossary_env(previous_env)\n'
        '\n'
        '    def _send_manga_glossary_page_image_batches(self, ocr_pages: List[Dict[str, Any]], client, temperature, max_tokens) -> Optional[List[Dict[str, Any]]]:\n'
        '        """Send the manga glossary requests with the page images ("Include page images in glossary generation").\n'
        '\n'
        '        The pages go in batches: the glossary Request Merging count when merging is on, otherwise one\n'
        '        page per request (the image glossary sends one image per request). A request is build_prompt()\n'
        "        over its pages' OCR text (_build_manga_glossary_input) with their images after it\n"
        '        (image_translator.image_glossary_page_parts: a page as its own file, a tall page cut into chunks\n'
        "        the way translation cuts it). The chunks of a tall page share the batch's request only with\n"
        "        SINGLE_API_IMAGE_CHUNKS=1; otherwise (the default) each chunk is its own request with its page's\n"
        '        OCR text, as the image glossary sends the chunks of a tall image.\n'
        '\n'
        '        Returns ``[{request, pages, images, response, entries}]`` (entries parsed, validated and\n'
        '        de-duplicated by the caller), or None when the model / route cannot take images: one warning,\n'
        '        and the caller sends the text-only request instead. A stop ends the run before the next request.\n'
        '        """\n'
        '        from extract_glossary_from_epub import build_prompt, parse_api_response\n'
        '        from image_translator import image_glossary_content, image_glossary_page_parts, is_image_input_unsupported_error\n'
        '        from TransateKRtoEN import send_with_interrupt\n'
        '        from unified_api_client import UnifiedClientError\n'
        '\n'
        "        config = getattr(self.main_gui, 'config', None) or {}\n"
        "        chunks_in_one_request = os.getenv('SINGLE_API_IMAGE_CHUNKS', '1') == '1'\n"
        '        pages_per_request = 1\n'
        "        if config.get('glossary_request_merging_enabled', False):\n"
        '            try:\n'
        "                pages_per_request = max(1, int(config.get('glossary_request_merge_count', 10) or 10))\n"
        '            except (TypeError, ValueError):\n'
        '                pages_per_request = 10\n'
        '        self._log(\n'
        '            f"🖼️ Manga glossary: sending the page images ({pages_per_request} page(s) per request, tall pages "\n'
        '            f"in chunks{\' in the same request\' if chunks_in_one_request else \', one request per chunk\'})", "info",\n'
        '        )\n'
        '\n'
        '        batches: List[Dict[str, Any]] = []\n'
        '        images_accepted = False\n'
        '        for start in range(0, len(ocr_pages), pages_per_request):\n'
        '            if self.stop_flag.is_set():\n'
        '                raise UnifiedClientError("Manga glossary generation stopped by user", error_type="cancelled")\n'
        '            whole_pages, whole_images, groups = [], [], []\n'
        '            for page in ocr_pages[start:start + pages_per_request]:\n'
        '                parts = image_glossary_page_parts(page.get(\'path\', \'\'), log=lambda message: self._log(message, "info"))\n'
        '                if len(parts) > 1 and not chunks_in_one_request:\n'
        '                    groups.extend(([page], [(page, label, url)]) for label, url in parts)\n'
        '                else:\n'
        '                    whole_pages.append(page)\n'
        '                    whole_images.extend((page, label, url) for label, url in parts)\n'
        '            if whole_pages:\n'
        '                groups.insert(0, (whole_pages, whole_images))\n'
        '            for group_pages, group_images in groups:\n'
        '                if self.stop_flag.is_set():\n'
        '                    raise UnifiedClientError("Manga glossary generation stopped by user", error_type="cancelled")\n'
        '                request_number = len(batches) + 1\n'
        '                system_prompt, user_prompt = build_prompt(self._build_manga_glossary_input(group_pages))\n'
        '                messages = [\n'
        '                    {"role": "system", "content": system_prompt},\n'
        '                    {"role": "user", "content": image_glossary_content(user_prompt, [url for _page, _label, url in group_images])},\n'
        '                ]\n'
        '                page_numbers = ", ".join(str(page.get(\'index\', \'?\')) for page in group_pages)\n'
        '                self._log(\n'
        '                    f"📤 Manga glossary request {request_number}: page(s) {page_numbers}, {len(group_images)} image(s)",\n'
        '                    "info",\n'
        '                )\n'
        '                try:\n'
        '                    response, _finish_reason, _raw_obj = send_with_interrupt(\n'
        '                        messages=messages,\n'
        '                        client=client,\n'
        '                        temperature=temperature,\n'
        '                        max_tokens=max_tokens,\n'
        '                        stop_check_fn=lambda: self.stop_flag.is_set(),\n'
        "                        context='glossary',\n"
        '                    )\n'
        '                except Exception as e:\n'
        '                    if group_images and not images_accepted and not self.stop_flag.is_set() and is_image_input_unsupported_error(e):\n'
        '                        self._log(f"⚠️ The model cannot take images ({e}); generating the manga glossary from the OCR text only", "warning")\n'
        '                        return None\n'
        '                    raise\n'
        '                images_accepted = images_accepted or bool(group_images)\n'
        '                response_text = self._normalize_manga_api_response_text(response)\n'
        '                self._log(f"📥 Manga glossary response {request_number} received ({len(response_text)} chars)", "info")\n'
        '                batches.append({\n'
        "                    'request': request_number,\n"
        "                    'pages': [\n"
        "                        {'index': page.get('index'), 'filename': os.path.basename(page.get('path', ''))}\n"
        '                        for page in group_pages\n'
        '                    ],\n'
        "                    'images': [\n"
        "                        {'page': page.get('index'), 'filename': os.path.basename(page.get('path', '')), 'image': label}\n"
        '                        for page, label, _url in group_images\n'
        '                    ],\n'
        "                    'response': response_text,\n"
        "                    'entries': parse_api_response(response_text) if response_text else [],\n"
        '                })\n'
        '        return batches\n'
    ),
    (
        '            "When enabled, the batch first runs OCR on every selected page, sends all OCR text in one glossary generation request, and then translates each page with that glossary appended to the translation prompt.\\n\\n"\n'
        '            "The Generate Glossary button runs the OCR and glossary generation step without translating pages."\n'
        '        ))\n'
        ,
        '            "When enabled, the batch first runs OCR on every selected page, sends all OCR text in one glossary generation request, and then translates each page with that glossary appended to the translation prompt.\\n\\n"\n'
        '            "The Generate Glossary button runs the OCR and glossary generation step without translating pages.\\n\\n"\n'
        '            "Include page images in glossary generation (on by default): the page images go with the OCR text, so the model "\n'
        '            "can see who is speaking, genders and what a name refers to, and correct OCR misreads. Tall pages are cut into "\n'
        '            "chunks the way image translation cuts them, one request per chunk (all in one request only in the Direct "\n'
        '            "image legacy mode). The pages are sent in batches (the glossary Request Merging count, or one page per "\n'
        '            "request when merging is off), and the entries of all requests are merged and de-duplicated. "\n'
        '            "A model that cannot take images falls back to the text-only request. Off: all OCR text in one text-only request."\n'
        '        ))\n'
    ),
    (
        '        manga_glossary_layout.addWidget(manga_glossary_toggle_frame)\n'
        '\n'
        ,
        '        manga_glossary_layout.addWidget(manga_glossary_toggle_frame)\n'
        '\n'
        '        manga_glossary_images_frame = QWidget()\n'
        '        manga_glossary_images_layout = QHBoxLayout(manga_glossary_images_frame)\n'
        '        manga_glossary_images_layout.setContentsMargins(20, 0, 0, 0)\n'
        '        manga_glossary_images_layout.setSpacing(10)\n'
        '\n'
        '        self.manga_glossary_include_images_checkbox = self._create_styled_checkbox("Include page images in glossary generation")\n'
        "        self.manga_glossary_include_images_checkbox.setChecked(bool(getattr(self, 'manga_glossary_include_images_value', self.main_gui.config.get('manga_glossary_include_images', True))))\n"
        '        self.manga_glossary_include_images_checkbox.setToolTip(\n'
        '            "Send the page images with the OCR text when generating the manga glossary (Generate Glossary\\n"\n'
        '            "and the glossary step of a batch run), so the model can see speakers, genders and what a name\\n"\n'
        '            "refers to, and correct OCR misreads. Tall pages are cut into chunks like image translation\\n"\n'
        '            "(one request per chunk unless the Direct image legacy mode sends them together).\\n"\n'
        '            "Pages per request: the glossary Request Merging count, or one page when merging is off.\\n"\n'
        '            "A model without image input falls back to text only. Off: one text-only request."\n'
        '        )\n'
        '        self.manga_glossary_include_images_checkbox.stateChanged.connect(self._on_manga_glossary_include_images_toggle)\n'
        '        manga_glossary_images_layout.addWidget(self.manga_glossary_include_images_checkbox)\n'
        '        manga_glossary_images_layout.addStretch()\n'
        '        manga_glossary_layout.addWidget(manga_glossary_images_frame)\n'
        '\n'
    ),
    (
        '        self.manga_glossary_enabled_value = enabled\n'
        '        self._save_rendering_settings()\n'
        ,
        '        self.manga_glossary_enabled_value = enabled\n'
        '        self._save_rendering_settings()\n'
        '\n'
        '    def _on_manga_glossary_include_images_toggle(self, state=None):\n'
        '        """Persist the manga glossary page-images toggle."""\n'
        '        try:\n'
        "            enabled = bool(self.manga_glossary_include_images_checkbox.isChecked()) if hasattr(self, 'manga_glossary_include_images_checkbox') else bool(state)\n"
        '        except Exception:\n'
        '            enabled = bool(state)\n'
        '        self.manga_glossary_include_images_value = enabled\n'
        '        self._save_rendering_settings()\n'
    ),
]


# 2026-11-06 the image progress tracker follows each run's output folder (PR 146 review): a second image or
# folder run of the same session used to write its translation_progress.json into the FIRST run's folder
_IMAGE_PROGRESS_FOLLOWS_RUN = (
    '                self.append_log(f"📊 Progress tracking in: {os.path.join(output_dir, \'translation_progress.json\')}")\n'
    '            \n'
    '            # Check for stop request early\n'
    ,
    '                self.append_log(f"📊 Progress tracking in: {os.path.join(output_dir, \'translation_progress.json\')}")\n'
    "            elif os.path.normcase(os.path.abspath(getattr(self.image_progress_manager, 'output_dir', None) or '')) != os.path.normcase(os.path.abspath(output_dir)):\n"
    '                # A later run (another image or folder) keeps its progress in its own output folder\n'
    '                os.makedirs(output_dir, exist_ok=True)\n'
    '                self.image_progress_manager.set_output_dir(output_dir)\n'
    '                self.append_log(f"📊 Progress tracking in: {os.path.join(output_dir, \'translation_progress.json\')}")\n'
    '            \n'
    '            # Check for stop request early\n'
)

# 2026-11-06 Image gen/edit key pool used everywhere: export_key_pool_env exports the Image gen/edit
# and Audio/TTS pool env like the other dedicated pools (translator_gui before U1, key_pools after);
# the prompt-only generation run sets both pools up like an input image and sends an image request
# with the 'image_generation' context (unified_api_client routes it to the Image gen/edit pool);
# image_job keeps that setup in one helper, ImageJobMixin._configure_media_output_key_pools.
def _media_pool_env(indent, config):
    pad = " " * indent
    return (pad + f"os.environ['TRUNCATION_RETRY_API_KEYS'] = json.dumps({config}.get('truncation_retry_keys', []))\n"
            + pad[4:] + "except Exception:\n",
            pad + f"os.environ['TRUNCATION_RETRY_API_KEYS'] = json.dumps({config}.get('truncation_retry_keys', []))\n"
            + pad + f"os.environ['USE_INPAINTER_KEYS'] = '1' if {config}.get('use_inpainter_keys', False) else '0'\n"
            + pad + f"os.environ['INPAINTER_API_KEYS'] = json.dumps({config}.get('inpainter_keys', []))\n"
            + pad + f"os.environ['USE_TTS_KEYS'] = '1' if {config}.get('use_tts_keys', False) else '0'\n"
            + pad + f"os.environ['TTS_API_KEYS'] = json.dumps({config}.get('tts_keys', []))\n"
            + pad[4:] + "except Exception:\n")


_MEDIA_POOL_ENV_TG = _media_pool_env(12, "self.config")   # TranslatorGUI._get_environment_variables (pre-U1)
_MEDIA_POOL_ENV_KEY_POOLS = _media_pool_env(8, "config")  # key_pools.export_key_pool_env

#: The Image gen/edit + Audio/TTS pool setup _process_image_file had inline (12-space indent).
_MEDIA_POOL_SETUP = (
    "            # Configure Image Gen/Edit key pool for image output mode requests.\n"
    "            try:\n"
    "                from unified_api_client import UnifiedClient\n"
    "                inpainter_keys_enabled = bool(self.config.get('use_inpainter_keys', False))\n"
    "                inpainter_keys = self.config.get('inpainter_keys', []) or []\n"
    "                os.environ['USE_INPAINTER_KEYS'] = '1' if inpainter_keys_enabled else '0'\n"
    "                os.environ['INPAINTER_API_KEYS'] = json.dumps(inpainter_keys)\n"
    "                if inpainter_keys_enabled and inpainter_keys:\n"
    "                    UnifiedClient.set_in_memory_inpainter_keys(\n"
    "                        inpainter_keys,\n"
    "                        force_rotation=self.config.get('force_key_rotation', True),\n"
    "                        rotation_frequency=self.config.get('rotation_frequency', 1),\n"
    "                    )\n"
    "                    self.append_log(f\"[ImageGenEdit] Key pool ENABLED for image output ({len(inpainter_keys)} keys)\")\n"
    "                else:\n"
    "                    UnifiedClient.clear_in_memory_inpainter_keys()\n"
    "                    if inpainter_keys_enabled:\n"
    "                        self.append_log(\"[ImageGenEdit] Enabled but no keys configured\")\n"
    "            except Exception:\n"
    "                pass\n"
    "\n"
    "            # Configure Audio / TTS key pool for audio output mode (context \"tts\").\n"
    "            try:\n"
    "                from unified_api_client import UnifiedClient\n"
    "                tts_keys_enabled = bool(self.config.get('use_tts_keys', False))\n"
    "                tts_keys = self.config.get('tts_keys', []) or []\n"
    "                os.environ['USE_TTS_KEYS'] = '1' if tts_keys_enabled else '0'\n"
    "                os.environ['TTS_API_KEYS'] = json.dumps(tts_keys)\n"
    "                if tts_keys_enabled and tts_keys:\n"
    "                    UnifiedClient.set_in_memory_tts_keys(\n"
    "                        tts_keys,\n"
    "                        force_rotation=self.config.get('force_key_rotation', True),\n"
    "                        rotation_frequency=self.config.get('rotation_frequency', 1),\n"
    "                    )\n"
    "                    self.append_log(f\"[AudioTTS] Key pool ENABLED for audio output ({len(tts_keys)} keys)\")\n"
    "                else:\n"
    "                    UnifiedClient.clear_in_memory_tts_keys()\n"
    "                    if tts_keys_enabled:\n"
    "                        self.append_log(\"[AudioTTS] Enabled but no keys configured\")\n"
    "            except Exception:\n"
    "                pass\n"
)
_MEDIA_POOL_CALL = "            self._configure_media_output_key_pools()\n"
_GENERATIVE_HEAD = ("            os.environ['NANOGPT_VIDEO_RESOLUTION'] = "
                    "str(getattr(self, 'nanogpt_video_resolution_var', '720p'))\n"
                    "\n")
_GENERATIVE_TAIL = "            # Instantiate a UnifiedClient for this call\n"
_GENERATIVE_POOLS_COMMENT = "            # Image gen/edit and Audio/TTS key pools, as for an input image\n"
_GENERATIVE_CONTEXT = (
    "            # An image request uses the Image gen/edit pool: unified_api_client routes the\n"
    "            # 'image_generation' context to it while USE_INPAINTER_KEYS is on. Video and audio keep\n"
    "            # the default context.\n"
    "            generates_image = os.environ['ENABLE_VIDEO_OUTPUT_MODE'] != '1' and (\n"
    "                os.environ['ENABLE_IMAGE_OUTPUT_MODE'] == '1' or self._model_is_image_gen(model))\n"
    "            send_kwargs = {'context': 'image_generation'} if generates_image else {}\n"
    "\n"
)


def _generative_pools(setup):
    return (_GENERATIVE_HEAD + _GENERATIVE_TAIL,
            _GENERATIVE_HEAD + _GENERATIVE_POOLS_COMMENT + setup + _GENERATIVE_CONTEXT + _GENERATIVE_TAIL)


_GENERATIVE_SEND = (
    "            result_text, _finish = client.send(\n"
    "                messages,\n"
    "                temperature=temperature,\n"
    "                max_tokens=int(getattr(self, 'max_output_tokens', 4096) or 4096),\n"
    "            )\n",
    "            result_text, _finish = client.send(\n"
    "                messages,\n"
    "                temperature=temperature,\n"
    "                max_tokens=int(getattr(self, 'max_output_tokens', 4096) or 4096),\n"
    "                **send_kwargs,\n"
    "            )\n",
)
#: image_job only: the runner pins (DEVFIX / U8) get the helper the current module has
_IMAGE_FILE_POOLS_CALL = (_MEDIA_POOL_SETUP,
                          "            # Configure the Image gen/edit and Audio/TTS key pools.\n" + _MEDIA_POOL_CALL)
_MEDIA_POOL_HELPER = (
    "    def _process_image_file(self, image_path, combined_output_dir=None):\n",
    "    def _configure_media_output_key_pools(self):\n"
    "        \"\"\"Set up the Image gen/edit and Audio/TTS key pools from the config (env + in-memory pools).\n"
    "\n"
    "        Both runners call it before their request: ``_process_image_file`` (where these lines\n"
    "        were) and ``_run_generative_prompt_mode``, which sent with the main key before.\n"
    "        \"\"\"\n"
    "        import json\n"
    "\n"
    + "".join((line[4:] if line else line) + "\n" for line in _MEDIA_POOL_SETUP.rstrip("\n").split("\n"))
    + "\n"
    "    def _process_image_file(self, image_path, combined_output_dir=None):\n",
)


# 2026-11-06 manga text requests ignore the main tab's Output mode: the editor's Translate / Translate
# This Text clients are text-only (UnifiedClient._text_output_only; the run's MangaTranslator marks its own)
_MANGA_TEXT_ONLY_EDITOR = [
    (
        '        client = UnifiedClient(model=model, api_key=api_key)\n'
        '        # If multi-key desired, ensure pool is initialized/refreshed\n'
        ,
        '        client = UnifiedClient(model=model, api_key=api_key)\n'
        "        client._text_output_only = True  # a text request: the main tab's Output mode is not applied\n"
        '        # If multi-key desired, ensure pool is initialized/refreshed\n'
    ),
    (
        '        # Create unified client with main GUI config\n'
        '        unified_client = UnifiedClient(model=model, api_key=api_key)\n'
        ,
        '        # Create unified client with main GUI config\n'
        '        unified_client = UnifiedClient(model=model, api_key=api_key)\n'
        "        unified_client._text_output_only = True  # a text request: the main tab's Output mode is not applied\n"
    ),
]
# 2026-11-06 Import OCR merge option ("Merge with detection", off by default): the imported regions join the
# page's detected / existing regions, an imported region replacing every one it overlaps
_OCR_IMPORT_MERGE = [
    (
        '        # Preserve provenance through the glossary workflow\'s OCR-only pass,\n'
        '        # which forwards these same objects as explicit precomputed regions.\n'
        '        for region in regions:\n'
        '            region._imported_ocr_session = True\n'
        '        return regions\n'
        ,
        '        # Preserve provenance through the glossary workflow\'s OCR-only pass,\n'
        '        # which forwards these same objects as explicit precomputed regions.\n'
        "        # Import OCR's merge option: the translator still detects the page and\n"
        '        # the imported regions join (replace overlapping) detected ones.\n'
        "        merge = bool(getattr(self, '_imported_ocr_merge', False))\n"
        '        for region in regions:\n'
        '            region._imported_ocr_session = True\n'
        '            region._imported_ocr_merge = merge\n'
        '        if merge and not regions:\n'
        '            return None  # nothing to merge: the page is detected as usual\n'
        '        return regions\n'
    ),
    (
        '        ocr_io_layout.addWidget(self.batch_ocr_import_btn, stretch=1)\n'
        '        ocr_io_layout.addWidget(self.batch_ocr_open_btn, stretch=1)\n'
        ,
        "        # Import OCR's merge option (this session's imports; off = the import replaces detection)\n"
        '        self.batch_ocr_import_merge_checkbox = self._create_styled_checkbox("Merge with detection")\n'
        "        self.batch_ocr_import_merge_checkbox.setChecked(bool(getattr(self, '_imported_ocr_merge', False)))\n"
        '        self.batch_ocr_import_merge_checkbox.setToolTip(\n'
        '            "Off: an imported page skips detection and uses only the imported regions.\\n"\n'
        '            "On: the page is still detected and the imported regions are added; an imported "\n'
        '            "region replaces every detected region it overlaps (for example an SFX-only OCR file "\n'
        '            "adds its sound effects and keeps the detected dialogue)."\n'
        '        )\n'
        '        self.batch_ocr_import_merge_checkbox.toggled.connect(\n'
        "            lambda checked: setattr(self, '_imported_ocr_merge', bool(checked)))\n"
        '        ocr_io_layout.addWidget(self.batch_ocr_import_btn, stretch=1)\n'
        '        ocr_io_layout.addWidget(self.batch_ocr_import_merge_checkbox)\n'
        '        ocr_io_layout.addWidget(self.batch_ocr_open_btn, stretch=1)\n'
    ),
    (
        '            if state_manager is not None:\n'
        '                for image_path, page in matches.items():\n'
        '                    imported_state = (\n'
        '                        imported_states.get(image_path)\n'
        '                        if imported_states is not None else None\n'
        '                    ) or manga_ocr_io.editor_state_from_page(page)\n'
        '                    existing_state = state_manager.get_state(image_path) or {}\n'
        '                    existing_state.update(imported_state)\n'
        ,
        '            if state_manager is not None:\n'
        "                merge = bool(getattr(self, '_imported_ocr_merge', False))\n"
        '                for image_path, page in matches.items():\n'
        '                    existing_state = state_manager.get_state(image_path) or {}\n'
        '                    if merge:\n'
        "                        # Merge with detection: the page's own boxes stay, except those\n"
        '                        # an imported region overlaps\n'
        '                        imported_state = manga_ocr_io.editor_state_for_import(page, existing_state, merge=True)\n'
        '                    else:\n'
        '                        imported_state = (\n'
        '                            imported_states.get(image_path)\n'
        '                            if imported_states is not None else None\n'
        '                        ) or manga_ocr_io.editor_state_from_page(page)\n'
        '                    existing_state.update(imported_state)\n'
    ),
]

# 2026-11-06 webtoon run (B1): the manga editor workflow matches the loaded glossary against the page's OCR
# text with the OCR settings (OCR drops most Korean word spaces; GLOSSARY_MATCH_HANGUL_SPACED=0)
_MANGA_GLOSSARY_OCR_MATCH_EDITOR = [
    (
        '    try:\n'
        '        from glossary_compressor import compress_glossary\n'
        '        glossary_path = ""\n'
        ,
        '    try:\n'
        '        from glossary_compressor import compress_glossary, OCR_SOURCE_MATCH_SETTINGS\n'
        '        glossary_path = ""\n'
    ),
    (
        '            },\n'
        '        )\n'
        ,
        '            },\n'
        '            # the page text is OCR output: its Korean word spacing is unreliable\n'
        '            settings=OCR_SOURCE_MATCH_SETTINGS,\n'
        '        )\n'
    ),
]


# 2026-11-06 webtoon run (B2): the built-in Manga_KR profile keeps Korean honorifics and family address terms
# like the Korean text profiles, and its quotation-mark line ends in real line breaks
_MANGA_KR_HONORIFICS = [
    (
        '                "- Keep speech patterns consistent with the character\'s appearance and demeanor.\\n"\n'
        '                "- Retain honorifics and onomatopoeia in Romaji.\\n"\n'
        '                "- Keep original Korean quotation marks (\\" \\", \' \', 「」, 『』) as-is without converting to English quotes.\\\\n\\\\n"\n'
        '\n'
        ,
        '                "- Keep speech patterns consistent with the character\'s appearance and demeanor.\\n"\n'
        '                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\\n"\n'
        '                "- Retain Korean familial address terms in romanized form rather than translating them (examples: oppa, eonni, hyung, noona, omma, appa, halabeoji, halmeoni), preserving their nuance and relationship context instead of converting them to English equivalents like brother, sister, mom, or dad.\\n"\n'
        '                "- Retain onomatopoeia in Romaji.\\n"\n'
        '                "- Keep original Korean quotation marks (\\" \\", \' \', 「」, 『』) as-is without converting to English quotes.\\n\\n"\n'
        '\n'
    ),
]


_MANGA_KR_OLD = (
    'You are a professional Korean to English Manhwa translator.\n'
    'You have both the image of the Manhwa panel and the extracted text to work with.\n'
    'Output only English text while following these rules: \n'
    '\n'
    'VISUAL CONTEXT:\n'
    '- Analyze the character’s facial expressions and body language in the image.\n'
    '- Consider the scene’s mood and atmosphere.\n'
    '- Note any action or movement depicted.\n'
    '- Use visual cues to determine the appropriate tone and emotion.\n'
    '- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n'
    '\n'
    'DIALOGUE REQUIREMENTS:\n'
    "- Match the translation tone to the character's expression.\n"
    '- If a character looks angry, use appropriately intense language.\n'
    '- If a character looks shy or embarrassed, reflect that in the translation.\n'
    "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
    '- Retain honorifics and onomatopoeia in Romaji.\n'
    '- Keep original Korean quotation marks (" ", \' \', 「」, 『』) as-is without converting to English quotes.\\n\\nIMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n'
)
_MANGA_KR_NEW = (
    'You are a professional Korean to English Manhwa translator.\n'
    'You have both the image of the Manhwa panel and the extracted text to work with.\n'
    'Output only English text while following these rules: \n'
    '\n'
    'VISUAL CONTEXT:\n'
    '- Analyze the character’s facial expressions and body language in the image.\n'
    '- Consider the scene’s mood and atmosphere.\n'
    '- Note any action or movement depicted.\n'
    '- Use visual cues to determine the appropriate tone and emotion.\n'
    '- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n'
    '\n'
    'DIALOGUE REQUIREMENTS:\n'
    "- Match the translation tone to the character's expression.\n"
    '- If a character looks angry, use appropriately intense language.\n'
    '- If a character looks shy or embarrassed, reflect that in the translation.\n'
    "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
    '- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n'
    '- Retain Korean familial address terms in romanized form rather than translating them (examples: oppa, eonni, hyung, noona, omma, appa, halabeoji, halmeoni), preserving their nuance and relationship context instead of converting them to English equivalents like brother, sister, mom, or dad.\n'
    '- Retain onomatopoeia in Romaji.\n'
    '- Keep original Korean quotation marks (" ", \' \', 「」, 『』) as-is without converting to English quotes.\n'
    '\n'
    'IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n'
)


# 2026-11-06 webtoon run (B3): the manga glossary prompt tells the model to correct OCR from the page image,
# fill gender from the art, keep titles used as names and the story's own names; saved old defaults migrated
_MANGA_GLOSSARY_IMAGE_NAME_RULES = [
    (
        '            "- Prefer recurring proper nouns and story-specific terms over ordinary dialogue words.\\n"\n'
        '            "- Keep raw_name exactly in the source script when possible."\n'
        '        )\n'
        ,
        '            "- Prefer recurring proper nouns and story-specific terms over ordinary dialogue words.\\n"\n'
        '            "- Keep raw_name exactly in the source script when possible.\\n"\n'
        '            "- Where these manga rules differ from the general rules above, follow these manga rules.\\n\\n"\n'
        '            "PAGE IMAGE RULES (when the page images are attached):\\n"\n'
        '            "- The OCR text is a machine reading and can be wrong: misread letters, dropped or merged words, missing spaces. "\n'
        '            "Check every name and term against the text printed in the image and write raw_name as it is printed there, "\n'
        '            "not as the OCR misread it. Never make an entry out of an OCR misreading, a cut-off fragment, or two words the OCR ran together.\\n"\n'
        '            "- raw_name is the name alone: leave out a particle or grammatical ending the OCR joined to it, and any honorific "\n'
        '            "(mention the honorific in the description).\\n"\n'
        '            "- Fill gender from the art: when the page shows the character, decide it from how they are drawn and how others "\n'
        '            "refer to them. Leave gender empty only when the character is never shown and the text gives no clue.\\n\\n"\n'
        '            "NAME RULES:\\n"\n'
        '            "- A title or epithet that the story uses instead of a name for one specific character (a character the cast "\n'
        '            "calls only by a rank, a role or a nickname) is a character entry, even though bare generic roles are rejected "\n'
        '            "above; its translated_name follows the same rules as any other name.\\n"\n'
        '            "- For places, organizations, skills, items and other named terms, translated_name is the name an English edition "\n'
        '            "would print: keep the story\'s own wording, puns and running jokes instead of glossing a compound name word by word, "\n'
        '            "and use the story\'s own spelling when the pages show one (a Latin-script sign, label or logo).\\n"\n'
        '            "- Keep honorifics and family address terms romanized (for example -nim, -ssi, unnie, oppa, -san, -sama, -kun, "\n'
        '            "shifu, gege) in translated_name and descriptions, instead of English titles such as Mr., Lady, Big Sister or Your Majesty."\n'
        '        )\n'
    ),
    (
        '        return f"{base_prompt}{manga_rules}"\n'
        '\n'
        ,
        '        return f"{base_prompt}{manga_rules}"\n'
        '\n'
        '    @staticmethod\n'
        '    def _migrate_legacy_manga_glossary_prompt(prompt, default_prompt):\n'
        '        """Add the page-image and name rules to a saved manga glossary prompt that still ends the way\n'
        '        the older default did (its last line was the raw_name rule).\n'
        '\n'
        '        2026-11-06 (webtoon run): the generator kept OCR misreadings as names, left most genders empty\n'
        "        with the page images on, skipped a title used as a character's name and glossed place and term\n"
        "        names word by word. A saved prompt with anything of the user's own after the old last line is\n"
        '        left alone."""\n'
        '        old_last_line = "- Keep raw_name exactly in the source script when possible."\n'
        '        if not isinstance(prompt, str) or not isinstance(default_prompt, str) or old_last_line not in default_prompt:\n'
        '            return prompt\n'
        '        added_rules = default_prompt[default_prompt.index(old_last_line) + len(old_last_line):]\n'
        '        if not added_rules or added_rules.strip() in prompt or not prompt.rstrip().endswith(old_last_line):\n'
        '            return prompt\n'
        '        return prompt.rstrip() + added_rules\n'
        '\n'
    ),
    (
        "        self.manga_glossary_prompt = config.get('manga_glossary_prompt', self._default_manga_glossary_prompt())\n"
        "        self.manga_custom_glossary_path = config.get('manga_custom_glossary_path', '')\n"
        ,
        "        self.manga_glossary_prompt = config.get('manga_glossary_prompt', self._default_manga_glossary_prompt())\n"
        '        migrated_glossary_prompt = self._migrate_legacy_manga_glossary_prompt(\n'
        '            self.manga_glossary_prompt, self._default_manga_glossary_prompt())\n'
        '        if migrated_glossary_prompt != self.manga_glossary_prompt:\n'
        '            self.manga_glossary_prompt = migrated_glossary_prompt\n'
        "            config['manga_glossary_prompt'] = migrated_glossary_prompt\n"
        '            print("[MANGA_INIT] Added the page-image and name rules to the saved manga glossary prompt")\n'
        "        self.manga_custom_glossary_path = config.get('manga_custom_glossary_path', '')\n"
    ),
]


_CONSOLE_WINDOW_MANGA = [
    (
        '        self.is_running = True\n'
        '        \n'
        ,
        '        self.is_running = True\n'
        '        self._set_console_visible(True)\n'
        '        \n'
    ),
    (
        '                    pass\n'
        '    \n'
        '    def _reset_ui_state(self, expected_start_token=None):\n'
        '        """Reset UI to ready state - with widget existence checks (PySide6)"""\n'
        ,
        '                    pass\n'
        '    \n'
        '    def _set_console_visible(self, running):\n'
        '        """Compiled builds: console_window shows the console while a manga run is active\n'
        '        (always when Other Settings > Show CMD is on)."""\n'
        '        try:\n'
        '            import console_window\n'
        '            console_window.set_manga_running(running)\n'
        '        except Exception:\n'
        '            pass\n'
        '\n'
        '    def _reset_ui_state(self, expected_start_token=None):\n'
        '        """Reset UI to ready state - with widget existence checks (PySide6)"""\n'
    ),
    (
        '        return False\n'
        '\n'
        '    def _reset_ui_state(self, expected_start_token=None):\n'
        '        """Reset UI to ready state - with widget existence checks (PySide6)"""\n'
        ,
        '        return False\n'
        '\n'
        '    def _set_console_visible(self, running):\n'
        '        """Compiled builds: console_window shows the console while a manga run is active\n'
        '        (always when Other Settings > Show CMD is on)."""\n'
        '        try:\n'
        '            import console_window\n'
        '            console_window.set_manga_running(running)\n'
        '        except Exception:\n'
        '            pass\n'
        '\n'
        '    def _reset_ui_state(self, expected_start_token=None):\n'
        '        """Reset UI to ready state - with widget existence checks (PySide6)"""\n'
    ),
    (
        '            self.is_running = False\n'
        '            self._graceful_stop_pending = False\n'
        ,
        '            self.is_running = False\n'
        '            self._set_console_visible(False)\n'
        '            self._graceful_stop_pending = False\n'
    ),
]


# 2026-11-07 image edit prompt profiles: the manga Custom Image Edit Prompt dialog gets the review dialog's
# shared profile row over manga_env.ImageEditPromptProfiles (Default, Translate SFX, Translate SFX +
# background text); boxes the user draws in the editor are marked user_drawn and saved with that mark
_IMAGE_EDIT_PROMPT_DIALOG_OLD = (
    '    def _edit_custom_image_edit_prompt(self):\n'
    '        """Open dialog to edit the prompt used by manga custom-image-edit inpainting."""\n'
    '        dialog = QDialog(self.dialog)\n'
    '        dialog.setWindowTitle("Custom Image Edit Prompt")\n'
    '        screen = QApplication.primaryScreen().geometry()\n'
    '        dialog.setMinimumSize(int(screen.width() * 0.42), int(screen.height() * 0.45))\n'
    '\n'
    '        layout = QVBoxLayout(dialog)\n'
    '        instructions = QLabel(\n'
    '            "Edit prompts sent only to the custom-image-edit inpainter request. "\n'
    '            "Use {target_lang} where the selected target language should be inserted."\n'
    '        )\n'
    '        instructions.setWordWrap(True)\n'
    '        layout.addWidget(instructions)\n'
    '\n'
    '        system_label = QLabel("System Prompt:")\n'
    '        layout.addWidget(system_label)\n'
    '        system_prompt_editor = QTextEdit()\n'
    '        system_prompt_editor.setPlainText(\n'
    "            getattr(self, 'custom_image_edit_system_prompt_value', '')\n"
    "            or self.main_gui.config.get('custom_image_edit_system_prompt', '')\n"
    "            or self.main_gui.config.get('custom_image_edit_prompt', '')\n"
    '            or self._default_custom_image_edit_system_prompt()\n'
    '        )\n'
    '        layout.addWidget(system_prompt_editor)\n'
    '\n'
    '        user_label = QLabel("User Prompt (optional):")\n'
    '        layout.addWidget(user_label)\n'
    '        user_prompt_editor = QTextEdit()\n'
    '        user_prompt_editor.setPlaceholderText("Optional extra instruction for this image edit request")\n'
    '        user_prompt_editor.setPlainText(\n'
    "            getattr(self, 'custom_image_edit_user_prompt_value', '')\n"
    "            or self.main_gui.config.get('custom_image_edit_user_prompt', '')\n"
    '        )\n'
    '        layout.addWidget(user_prompt_editor)\n'
    '\n'
    '        button_layout = QHBoxLayout()\n'
    '\n'
    '        def save_prompt():\n'
    '            system_prompt = system_prompt_editor.toPlainText().strip() or self._default_custom_image_edit_system_prompt()\n'
    '            user_prompt = user_prompt_editor.toPlainText().strip()\n'
    "            self._sync_custom_image_edit_prompt(system_prompt=system_prompt, user_prompt=user_prompt, source='manga')\n"
    '            self._save_rendering_settings()\n'
    '            self._log("Updated custom image edit prompts", "success")\n'
    '            dialog.accept()\n'
    '\n'
    '        def reset_prompt():\n'
    '            reply = QMessageBox.question(\n'
    '                dialog,\n'
    '                "Reset Custom Image Edit Prompts",\n'
    '                "Reset the custom image edit system prompt to the default and clear the optional user prompt?",\n'
    '                QMessageBox.Yes | QMessageBox.No,\n'
    '                QMessageBox.No\n'
    '            )\n'
    '            if reply != QMessageBox.Yes:\n'
    '                return\n'
    '            system_prompt_editor.setPlainText(self._default_custom_image_edit_system_prompt())\n'
    '            user_prompt_editor.clear()\n'
    '\n'
    '        save_btn = QPushButton("Save")\n'
    '        save_btn.clicked.connect(save_prompt)\n'
    '        button_layout.addWidget(save_btn)\n'
    '\n'
    '        reset_btn = QPushButton("Reset to Default")\n'
    '        reset_btn.clicked.connect(reset_prompt)\n'
    '        button_layout.addWidget(reset_btn)\n'
    '\n'
    '        cancel_btn = QPushButton("Cancel")\n'
    '        cancel_btn.clicked.connect(dialog.reject)\n'
    '        button_layout.addWidget(cancel_btn)\n'
    '        button_layout.addStretch()\n'
    '\n'
    '        layout.addLayout(button_layout)\n'
    '        dialog.exec()\n'
    '\n'
)
_IMAGE_EDIT_PROMPT_DIALOG_NEW = (
    '    def _edit_custom_image_edit_prompt(self):\n'
    '        """Open dialog to edit the prompt used by manga custom-image-edit inpainting.\n'
    '\n'
    "        A profile row like the review / glossary prompt profiles (review_dialog's shared row over\n"
    '        manga_env.ImageEditPromptProfiles): Default (the prompt used before profiles), the built-in\n'
    "        Translate SFX and Translate SFX + background text, and the user's own profiles. The selected\n"
    '        profile\'s prompts are the ones the image edit requests send."""\n'
    '        from review_dialog import build_prompt_profile_row, refresh_prompt_profile_row, confirm_prompt_profile_action\n'
    '        from manga_env import ImageEditPromptProfiles\n'
    '\n'
    '        dialog = QDialog(self.dialog)\n'
    '        dialog.setWindowTitle("Custom Image Edit Prompt")\n'
    '        screen = QApplication.primaryScreen().geometry()\n'
    '        dialog.setMinimumSize(int(screen.width() * 0.42), int(screen.height() * 0.45))\n'
    '\n'
    '        layout = QVBoxLayout(dialog)\n'
    '        instructions = QLabel(\n'
    '            "Edit prompts sent only to the custom-image-edit inpainter request. "\n'
    '            "Use {target_lang} where the selected target language should be inserted. "\n'
    '            "A prompt with {target_lang} asks the model to draw translated text (Translate SFX profiles): "\n'
    '            "the whole crop is sent, not only the detected text boxes, so sound effects and "\n'
    '            "background text the detector missed are translated too. Only what the model redrew is kept "\n'
    '            "(the source art elsewhere), and Glossarion letters only the speech bubbles and captions."\n'
    '        )\n'
    '        instructions.setWordWrap(True)\n'
    '        layout.addWidget(instructions)\n'
    '\n'
    '        system_label = QLabel("System Prompt:")\n'
    '        system_prompt_editor = QTextEdit()\n'
    '        user_label = QLabel("User Prompt (optional):")\n'
    '        user_prompt_editor = QTextEdit()\n'
    '        user_prompt_editor.setPlaceholderText("Optional extra instruction for this image edit request")\n'
    '\n'
    '        def persist():\n'
    '            try:\n'
    '                return self.main_gui.save_config(show_message=False)\n'
    '            except Exception:\n'
    '                return False\n'
    '\n'
    '        def write_runtime(texts):\n'
    "            # The selected profile's prompts are what the runs read (tab values, main GUI vars, config).\n"
    '            self._sync_custom_image_edit_prompt(\n'
    "                system_prompt=texts.get('system', ''), user_prompt=texts.get('user', ''), source='save')\n"
    '\n'
    '        profiles = ImageEditPromptProfiles(\n'
    '            self.main_gui, persist=persist, log=lambda message: self._log(message, "info"), on_runtime=write_runtime)\n'
    '        dialog._image_edit_prompt_profiles = profiles\n'
    '\n'
    '        def editor_texts():\n'
    "            return {'system': system_prompt_editor.toPlainText(), 'user': user_prompt_editor.toPlainText()}\n"
    '\n'
    '        def show_selected():\n'
    '            texts = profiles.texts\n'
    "            system_prompt_editor.setPlainText(texts.get('system') or self._default_custom_image_edit_system_prompt())\n"
    "            user_prompt_editor.setPlainText(texts.get('user', ''))\n"
    '            refresh_prompt_profile_row(\n'
    '                combo, buttons, profiles,\n'
    '                "Restore the built-in \'{name}\' image edit prompts", "Delete the selected image edit prompt profile")\n'
    '\n'
    '        def show_box(box):\n'
    '            if box:\n'
    '                QMessageBox.warning(dialog, box[0], box[1])\n'
    '            return box\n'
    '\n'
    '        def select_profile(name):\n'
    "            name = str(name or '').strip()\n"
    '            if not name or name == profiles.selected_name() or profiles.profile_texts(name) is None:\n'
    '                return\n'
    '            # Unsaved edits stay with the profile they were made in.\n'
    '            profiles.stage(profiles.selected_name(), editor_texts())\n'
    '            if profiles.select(name) is None:\n'
    '                return\n'
    '            show_selected()\n'
    '            persist()\n'
    '\n'
    '        def new_profile():\n'
    '            profiles.stage(profiles.selected_name(), editor_texts())\n'
    '            box = profiles.new()\n'
    '            show_selected()\n'
    '            show_box(box)\n'
    '\n'
    '        def save_profile():\n'
    '            box = profiles.save(combo.currentText(), editor_texts())\n'
    '            if show_box(box):\n'
    '                return False\n'
    '            show_selected()\n'
    '            self._save_rendering_settings()\n'
    '            self._log("Updated custom image edit prompts", "success")\n'
    '            return True\n'
    '\n'
    '        def delete_or_reset_profile():\n'
    '            name = profiles.selected_name()\n'
    '            if profiles.is_builtin(name):\n'
    '                if not confirm_prompt_profile_action(\n'
    '                        dialog, "Reset Profile",\n'
    '                        f"Reset the built-in \'{name}\' image edit prompt profile to its shipped prompts?"):\n'
    '                    return\n'
    '                box = profiles.reset(name)\n'
    '            else:\n'
    '                box = profiles.delete(name, confirm=lambda n: confirm_prompt_profile_action(\n'
    '                    dialog, "Delete Profile", f"Delete image edit prompt profile \'{n}\'?"))\n'
    '            show_selected()\n'
    '            show_box(box)\n'
    '\n'
    '        row, combo, buttons = build_prompt_profile_row(\n'
    '            "Profile:", "image_edit",\n'
    '            "Select an image edit prompt profile, or edit its name and save to rename it.\\n"\n'
    '            "Each profile stores the system prompt and the optional user prompt.",\n'
    "            {'new': new_profile, 'save': save_profile, 'delete': delete_or_reset_profile},\n"
    '            select_profile)\n'
    '        dialog._image_edit_profile_combo = combo\n'
    '        dialog._image_edit_profile_buttons = buttons\n'
    '        layout.addWidget(row)\n'
    '        layout.addWidget(system_label)\n'
    '        layout.addWidget(system_prompt_editor)\n'
    '        layout.addWidget(user_label)\n'
    '        layout.addWidget(user_prompt_editor)\n'
    '        dialog._image_edit_prompt_editors = (system_prompt_editor, user_prompt_editor)\n'
    '        show_selected()\n'
    '\n'
    '        button_layout = QHBoxLayout()\n'
    '\n'
    '        def save_prompt():\n'
    '            if save_profile():\n'
    '                dialog.accept()\n'
    '\n'
    '        def reset_prompt():\n'
    '            name = profiles.selected_name()\n'
    "            shipped = profiles.builtin_texts(name) or profiles.builtin_texts('Default')\n"
    '            reply = QMessageBox.question(\n'
    '                dialog,\n'
    '                "Reset Custom Image Edit Prompts",\n'
    '                f"Reset the editors to the shipped prompts of \'{name}\'"\n'
    '                + ("" if profiles.is_builtin(name) else " (the default prompt)") + "? Save to keep them.",\n'
    '                QMessageBox.Yes | QMessageBox.No,\n'
    '                QMessageBox.No\n'
    '            )\n'
    '            if reply != QMessageBox.Yes:\n'
    '                return\n'
    "            system_prompt_editor.setPlainText(shipped.get('system', '') or self._default_custom_image_edit_system_prompt())\n"
    "            user_prompt_editor.setPlainText(shipped.get('user', ''))\n"
    '\n'
    '        save_btn = QPushButton("Save")\n'
    '        save_btn.clicked.connect(save_prompt)\n'
    '        button_layout.addWidget(save_btn)\n'
    '\n'
    '        reset_btn = QPushButton("Reset to Default")\n'
    '        reset_btn.clicked.connect(reset_prompt)\n'
    '        button_layout.addWidget(reset_btn)\n'
    '\n'
    '        cancel_btn = QPushButton("Cancel")\n'
    '        cancel_btn.clicked.connect(dialog.reject)\n'
    '        button_layout.addWidget(cancel_btn)\n'
    '        button_layout.addStretch()\n'
    '\n'
    '        layout.addLayout(button_layout)\n'
    '        dialog.exec()\n'
    '\n'
)
_IMAGE_EDIT_PROMPT_PROFILES = [(_IMAGE_EDIT_PROMPT_DIALOG_OLD, _IMAGE_EDIT_PROMPT_DIALOG_NEW)]
_USER_DRAWN_BOXES_PREVIEW = [
    (
        '                rect_item.is_recognized = False\n'
        '            except Exception:\n'
        '                pass\n'
        '            # Attach context menu via integration if available\n',
        '                rect_item.is_recognized = False\n'
        '                # Drawn by the user: cleaned / image-edited even when OCR reads no text in it\n'
        '                rect_item.user_drawn = True\n'
        '            except Exception:\n'
        '                pass\n'
        '            # Attach context menu via integration if available\n',
    ),
    ("                    for key in ('bubble_type', 'region_type', 'bubble_bounds'):\n",
     "                    for key in ('bubble_type', 'region_type', 'bubble_bounds', 'user_drawn'):\n"),
]


# 2026-11-07: Inpainting > Inpaint Method: Replicate (Bad) / Local (Light-weight) / API (Recommended) /
# Hybrid (Trash). API is the custom-image-edit inpainter (manga_inpaint_method 'api'); the run path treats
# 'api' like 'local' + 'custom-image-edit'. Pairs generated from the real diff hunks (editor functions
# frozen in ImageRenderer, the batch runner's worker frozen in manga_integration).
_INPAINT_API_MANGA_INTEGRATION = [
    ('    _preload_completed_models = set()  # Track which models have been loaded\n    \n    def __init__(self, parent_widget, main_gui, dialog, scroll_area=None):\n',
     "    _preload_completed_models = set()  # Track which models have been loaded\n\n    # Inpainting > Local (Light-weight) model dropdown. 'custom-image-edit' is not listed: it is\n    # the API (Recommended) method (manga_inpaint_method 'api').\n    LOCAL_INPAINT_MODEL_CHOICES = ('aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'mat', 'ollama', 'sd_local')\n\n    def __init__(self, parent_widget, main_gui, dialog, scroll_area=None):\n"),
    ("        inpainting_method = inpaint_settings.get('method', 'local')\n        return inpainting_method == 'local'\n\n",
     "        inpainting_method = inpaint_settings.get('method', 'local')\n        return inpainting_method in ('local', 'api')  # 'api' = the custom-image-edit inpainter\n\n"),
    ("        inpainting_method = inpaint_settings.get('method', 'local')\n        inpainting_enabled = not skip_inpainting and inpainting_method == 'local'\n        \n",
     "        inpainting_method = inpaint_settings.get('method', 'local')\n        inpainting_enabled = not skip_inpainting and inpainting_method in ('local', 'api')\n        \n"),
    ("                # Check top-level config first (manga_local_inpaint_model), then nested config\n                local_method = self.main_gui.config.get('manga_local_inpaint_model', \n                                                        inpaint_settings.get('local_method', 'anime_onnx'))\n",
     "                # Check top-level config first (manga_local_inpaint_model), then nested config\n                local_method = self.main_gui.config.get('manga_local_inpaint_model',\n                                                        inpaint_settings.get('local_method', 'anime_onnx'))\n"),
    ("                                                        inpaint_settings.get('local_method', 'anime_onnx'))\n                model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')\n",
     "                                                        inpaint_settings.get('local_method', 'anime_onnx'))\n                if inpainting_method == 'api':\n                    local_method = 'custom-image-edit'\n                model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')\n"),
    ("\n        self.inpaint_method_value = self.main_gui.config.get('manga_inpaint_method', 'local')\n        self.inpaint_method_group = QButtonGroup()\n",
     "\n        # The custom image edit endpoint used to be the 'local' method with the local model\n        # 'custom-image-edit'; such configs move to the 'api' method here (saved with the next save).\n        from manga_models import migrate_inpaint_method, normalize_inpaint_method\n        migrate_inpaint_method(self.main_gui.config)\n        self.inpaint_method_value = normalize_inpaint_method(\n            self.main_gui.config.get('manga_inpaint_method', 'local'),\n            self.main_gui.config.get('manga_local_inpaint_model', ''),\n        )\n        self.inpaint_method_group = QButtonGroup()\n"),
    ('        radio_font = QFont(\'Arial\', 9)\n        \n        cloud_radio = QRadioButton("Replicate API")\n        cloud_radio.setFont(radio_font)\n',
     '        radio_font = QFont(\'Arial\', 9)\n\n        cloud_radio = QRadioButton("Replicate (Bad)")\n        cloud_radio.setFont(radio_font)\n'),
    ('\n        local_radio = QRadioButton("Local / API Model")\n        local_radio.setFont(radio_font)\n',
     '\n        local_radio = QRadioButton("Local (Light-weight)")\n        local_radio.setFont(radio_font)\n'),
    ('        local_radio.setToolTip(\n            "For best local performance, select anime_onnx from the model dropdown.\\n"\n            "For best results, select custom-image-edit and use a modern image edit model.\\n"\n            "Recommended for image edit models such as Nano Banana 2 and Wan 2.6 Image Edit."\n        )\n',
     '        local_radio.setToolTip(\n            "Run a local inpainting model on this computer.\\n"\n            "For best local performance, select anime_onnx from the model dropdown."\n        )\n'),
    ('\n        hybrid_radio = QRadioButton("Hybrid")\n        hybrid_radio.setFont(radio_font)\n',
     '\n        api_radio = QRadioButton("API (Recommended)")\n        api_radio.setFont(radio_font)\n        api_radio.setToolTip(\n            "Clean the text with an image edit model through the custom image edit endpoint.\\n"\n            "Recommended for image edit models such as Nano Banana 2 and Wan 2.6 Image Edit."\n        )\n        api_radio.setChecked(self.inpaint_method_value == \'api\')\n        api_radio.toggled.connect(lambda checked: self._on_inpaint_method_change() if checked else None)\n        self.inpaint_method_group.addButton(api_radio, 3)\n        method_selection_layout.addWidget(api_radio)\n\n        hybrid_radio = QRadioButton("Hybrid (Trash)")\n        hybrid_radio.setFont(radio_font)\n'),
    ('        hybrid_radio.setToolTip(\n            "Show and use both cloud and local/API inpainting settings.\\n"\n            "Experimental and not recommended for normal use."\n',
     '        hybrid_radio.setToolTip(\n            "Show and use both cloud and local inpainting settings.\\n"\n            "Experimental and not recommended for normal use."\n'),
    ('        self.local_radio = local_radio\n        self.hybrid_radio = hybrid_radio\n',
     '        self.local_radio = local_radio\n        self.api_radio = api_radio\n        self.hybrid_radio = hybrid_radio\n'),
    ("                    self.main_gui.config['manga_custom-image-edit_model_path'] = endpoint\n        local_model_combo = QComboBox()\n",
     "                    self.main_gui.config['manga_custom-image-edit_model_path'] = endpoint\n        # The Local radio lists the local inpainting models only; the API radio is the\n        # custom-image-edit inpainter and has no dropdown.\n        local_choices = list(self.LOCAL_INPAINT_MODEL_CHOICES)\n        if self.local_model_type_value not in local_choices:\n            self.local_model_type_value = 'anime_onnx'\n        local_model_choice = self.local_model_type_value\n        if self.inpaint_method_value == 'api':\n            self.local_model_type_value = 'custom-image-edit'\n        local_model_combo = QComboBox()\n"),
    ("        local_model_combo = QComboBox()\n        local_model_combo.addItems(['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'custom-image-edit', 'mat', 'ollama', 'sd_local'])\n        local_model_combo.setCurrentText(self.local_model_type_value)\n        local_model_combo.setMinimumWidth(150)\n",
     '        local_model_combo = QComboBox()\n        local_model_combo.addItems(local_choices)\n        local_model_combo.setCurrentText(local_model_choice)\n        local_model_combo.setMinimumWidth(150)\n'),
    ("        }\n        self.model_desc_label = QLabel(model_desc.get(self.local_model_type_value, ''))\n        desc_font = QFont('Arial', 8)\n",
     "        }\n        self.model_desc_label = QLabel(model_desc.get(local_model_choice, ''))\n        desc_font = QFont('Arial', 8)\n"),
    ('        local_model_layout.addStretch()\n        \n        local_inpaint_layout.addWidget(local_model_frame)\n',
     '        local_model_layout.addStretch()\n\n        local_inpaint_layout.addWidget(local_model_frame)\n'),
    ('        local_inpaint_layout.addWidget(local_model_frame)\n\n',
     '        local_inpaint_layout.addWidget(local_model_frame)\n        self.local_model_frame = local_model_frame\n\n'),
    ('        local_inpaint_layout.addWidget(model_info_btn)\n        \n        # Add local_inpaint_frame to inpaint_group\n',
     '        local_inpaint_layout.addWidget(model_info_btn)\n        self.model_info_btn = model_info_btn\n\n        # Add local_inpaint_frame to inpaint_group\n'),
    ("                        method = str(getattr(self, 'inpaint_method_value', '') or '').lower()\n                        if method == 'local':\n                            self._trigger_inpainter_preload_after_toggle()\n",
     "                        method = str(getattr(self, 'inpaint_method_value', '') or '').lower()\n                        if method in ('local', 'api'):\n                            self._trigger_inpainter_preload_after_toggle()\n"),
    ("            method = 'local'\n        elif self.hybrid_radio.isChecked():\n",
     "            method = 'local'\n        elif getattr(self, 'api_radio', None) is not None and self.api_radio.isChecked():\n            method = 'api'\n        elif self.hybrid_radio.isChecked():\n"),
    ("            method = 'local'  # Default fallback\n        \n        # Update the stored value\n",
     "            method = 'local'  # Default fallback\n\n        # Update the stored value\n"),
    ('        self.inpaint_method_value = method\n        \n        # Show/hide frames based on method\n',
     "        self.inpaint_method_value = method\n\n        # API (Recommended) is the custom-image-edit inpainter; the other methods use the\n        # Local Model dropdown's model.\n        previous_model = str(getattr(self, 'local_model_type_value', '') or '')\n        if method == 'api':\n            model = 'custom-image-edit'\n        else:\n            model = self.local_model_combo.currentText() if hasattr(self, 'local_model_combo') else previous_model\n\n        # Show/hide frames based on method\n"),
    ("        # Cloud frame: visible for cloud and hybrid\n        # Local frame: visible for local and hybrid\n        if method == 'cloud':\n",
     "        # Cloud frame: visible for cloud and hybrid\n        # Local frame: visible for local, api and hybrid (api hides the model dropdown)\n        if method == 'cloud':\n"),
    ("            self.local_inpaint_frame.hide()\n        elif method == 'local':\n            self.cloud_inpaint_frame.hide()\n",
     "            self.local_inpaint_frame.hide()\n        elif method in ('local', 'api'):\n            self.cloud_inpaint_frame.hide()\n"),
    ('            self.local_inpaint_frame.show()\n        \n',
     "            self.local_inpaint_frame.show()\n        if hasattr(self, 'local_model_frame'):\n            self.local_model_frame.setVisible(method != 'api')\n        if hasattr(self, 'model_info_btn'):\n            self.model_info_btn.setVisible(method != 'api')\n\n        if model and model != previous_model:\n            if getattr(self, '_initializing', False) or method == 'cloud':\n                # nothing to load: Replicate does not use the local model\n                self.local_model_type_value = model\n                self._apply_custom_image_edit_ui_state()\n            else:\n                # Same path as picking the model in the dropdown: load / endpoint state, then save\n                self.local_model_type_value = model\n                self._on_local_model_change(model)\n                return\n        \n"),
]

_INPAINT_API_RUNNER = [
    ("                    effective_workers > 1\n                    and inpaint_method == 'local'\n                    and hasattr(self, 'translator')\n",
     "                    effective_workers > 1\n                    and inpaint_method in ('local', 'api')\n                    and hasattr(self, 'translator')\n"),
    ("                    local_method = self.main_gui.config.get('manga_local_inpaint_model', 'anime')\n                    model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')\n",
     "                    local_method = self.main_gui.config.get('manga_local_inpaint_model', 'anime')\n                    if inpaint_method == 'api':  # API (Recommended) = the custom-image-edit inpainter\n                        local_method = 'custom-image-edit'\n                    model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')\n"),
]

_INPAINT_API_EDITOR = [
    ("        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')\n        is_custom_image_edit = inpaint_method == 'local' and str(local_model or '').lower() == 'custom-image-edit'\n        \n",
     "        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')\n        if inpaint_method == 'api':  # API (Recommended) = the custom-image-edit inpainter\n            local_model = 'custom-image-edit'\n        is_custom_image_edit = inpaint_method in ('local', 'api') and str(local_model or '').lower() == 'custom-image-edit'\n        \n"),
    ("        \n        if inpaint_method == 'local':\n            # Use local inpainter with the same method as manga_translator\n",
     "        \n        if inpaint_method in ('local', 'api'):\n            # Use local inpainter with the same method as manga_translator\n"),
    ("        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')\n        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')\n        \n        is_custom_image_edit = False\n",
     "        inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'local')\n        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')\n        if inpaint_method == 'api':  # API (Recommended) = the custom-image-edit inpainter\n            local_model = 'custom-image-edit'\n        \n        is_custom_image_edit = False\n"),
    ("        is_custom_image_edit = False\n        if inpaint_method == 'local':\n            # Use local inpainter with the same method as manga_translator\n",
     "        is_custom_image_edit = False\n        if inpaint_method in ('local', 'api'):\n            # Use local inpainter with the same method as manga_translator\n"),
    ("        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')\n\n",
     "        local_model = self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx')\n        if inpaint_method == 'api':  # API (Recommended) = the custom-image-edit inpainter\n            local_model = 'custom-image-edit'\n\n"),
    ('\n        if inpaint_method != \'local\':\n            print(f"[PRELOAD_INPAINTER] Skipping preload - method is {inpaint_method}, not local")\n',
     '\n        if inpaint_method not in (\'local\', \'api\'):\n            print(f"[PRELOAD_INPAINTER] Skipping preload - method is {inpaint_method}, not local")\n'),
    ('        inpaint_method = self.main_gui.config.get(\'manga_inpaint_method\', \'local\')\n        local_model = self.main_gui.config.get(\'manga_local_inpaint_model\', \'anime_onnx\')\n        \n        print(f"[INPAINT_REGION] Using method: {inpaint_method}, model: {local_model}")\n',
     '        inpaint_method = self.main_gui.config.get(\'manga_inpaint_method\', \'local\')\n        local_model = self.main_gui.config.get(\'manga_local_inpaint_model\', \'anime_onnx\')\n        if inpaint_method == \'api\':  # API (Recommended) = the custom-image-edit inpainter\n            local_model = \'custom-image-edit\'\n        \n        print(f"[INPAINT_REGION] Using method: {inpaint_method}, model: {local_model}")\n'),
    ("        \n        if inpaint_method == 'local':\n            # Get model path from config (same way as _run_clean_background)\n",
     "        \n        if inpaint_method in ('local', 'api'):\n            # Get model path from config (same way as _run_clean_background)\n"),
    ('                        self._cleaned_image_path = None\n                    # Only run inpainting if method is not \'none\' and is \'local\' or \'hybrid\'\n                    elif inpaint_method in [\'local\', \'hybrid\']:\n                        self._log(f"🧹 [{idx}/{total}] Cleaning image...", "info")\n',
     '                        self._cleaned_image_path = None\n                    # Only run inpainting if method is not \'none\' and is \'local\', \'api\' or \'hybrid\'\n                    elif inpaint_method in [\'local\', \'api\', \'hybrid\']:\n                        self._log(f"🧹 [{idx}/{total}] Cleaning image...", "info")\n'),
]


_DEL_KEY_IMPORT_EXPORT_PREVIEW = [
    (
        '        self.import_ocr_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)\n'
        ,
        "        # Fill the toolbar row's height instead of sitting short in it\n"
        '        self.import_ocr_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)\n'
    ),
    (
        '        self.export_ocr_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)\n'
        ,
        "        # Fill the toolbar row's height instead of sitting short in it\n"
        '        self.export_ocr_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)\n'
    ),
    (
        '        # Forward left/right arrow keys to parent for image navigation\n'
        '        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):\n'
        ,
        '        # Forward left/right arrow keys to parent for image navigation; Delete too (deletes the\n'
        "        # selected box there), unless a box's text is being typed into\n"
        '        scene = self.scene()\n'
        '        typing = scene is not None and scene.focusItem() is not None\n'
        '        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right) or (event.key() == Qt.Key.Key_Delete and not typing):\n'
    ),
    (
        '            self._navigate_to_next_image()\n'
        '            event.accept()\n'
        '            return\n'
        '        \n'
        '        # Pass other keys to parent\n'
        ,
        '            self._navigate_to_next_image()\n'
        '            event.accept()\n'
        '            return\n'
        '\n'
        "        # Delete key - the 🗑 Delete Selected button's action (shown while manual editing is on)\n"
        '        elif event.key() == Qt.Key.Key_Delete and self.manual_editing_enabled:\n'
        '            self.viewer.delete_selected_rectangle()\n'
        '            event.accept()\n'
        '            return\n'
        '        \n'
        '        # Pass other keys to parent\n'
    ),
]


# 2026-11-13 Relations Tracker (glossary_enable_relations_tracker / GLOSSARY_ENABLE_RELATIONS_TRACKER,
# glossary_relations_history_mode / GLOSSARY_RELATIONS_HISTORY_MODE, glossary_relations_field_name /
# GLOSSARY_RELATIONS_FIELD): each pair is (row before + row after) -> (before + new rows + after), so it
# applies once at every pin and is a no-op where the rows already exist.
_RELATIONS_TRACKER_ROWS = {'src/owner_state.py': [('        self.glossary_gender_tracking_bias_var = '
                         "self.config.get('glossary_gender_tracking_bias', 'none')\n"
                         '        self.glossary_partial_ratio_gender_only_var = '
                         "self.config.get('glossary_partial_ratio_gender_only', False)\n",
                         '        self.glossary_gender_tracking_bias_var = '
                         "self.config.get('glossary_gender_tracking_bias', 'none')\n"
                         '        self.glossary_enable_relations_tracker_var = '
                         "self.config.get('glossary_enable_relations_tracker', False)\n"
                         '        self.glossary_relations_history_mode_var = '
                         "self.config.get('glossary_relations_history_mode', 'previous')\n"
                         '        self.glossary_partial_ratio_gender_only_var = '
                         "self.config.get('glossary_partial_ratio_gender_only', False)\n")],
 'src/run_env.py': [("            'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, "
                     "'glossary_gender_tracking_bias_var', self.config.get('glossary_gender_tracking_bias', "
                     "'none'))),\n"
                     "            'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, "
                     "'glossary_partial_ratio_gender_only_var', False) else '0',\n",
                     "            'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, "
                     "'glossary_gender_tracking_bias_var', self.config.get('glossary_gender_tracking_bias', "
                     "'none'))),\n"
                     "            'GLOSSARY_ENABLE_RELATIONS_TRACKER': '1' if getattr(self, "
                     "'glossary_enable_relations_tracker_var', False) else '0',\n"
                     "            'GLOSSARY_RELATIONS_HISTORY_MODE': str(getattr(self, "
                     "'glossary_relations_history_mode_var', "
                     "self.config.get('glossary_relations_history_mode', 'previous'))),\n"
                     "            'GLOSSARY_RELATIONS_FIELD': "
                     "str(self.config.get('glossary_relations_field_name', 'relations')),\n"
                     "            'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, "
                     "'glossary_partial_ratio_gender_only_var', False) else '0',\n"),
                    ("            ('GLOSSARY_GENDER_TRACKING_BIAS', "
                     "str(self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                     "            ('GLOSSARY_USE_SMART_FILTER', '1' if "
                     "self.config.get('glossary_use_smart_filter', True) else '0'),\n",
                     "            ('GLOSSARY_GENDER_TRACKING_BIAS', "
                     "str(self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                     "            ('GLOSSARY_ENABLE_RELATIONS_TRACKER', '1' if "
                     "self.config.get('glossary_enable_relations_tracker', False) else '0'),\n"
                     "            ('GLOSSARY_RELATIONS_HISTORY_MODE', "
                     "str(self.config.get('glossary_relations_history_mode', 'previous'))),\n"
                     "            ('GLOSSARY_RELATIONS_FIELD', "
                     "str(self.config.get('glossary_relations_field_name', 'relations'))),\n"
                     "            ('GLOSSARY_USE_SMART_FILTER', '1' if "
                     "self.config.get('glossary_use_smart_filter', True) else '0'),\n")],
 'src/settings_persistence.py': [("            ('glossary_gender_tracking_bias', "
                                  "['glossary_gender_tracking_bias_var'], 'none', str),\n"
                                  "            ('glossary_filter_mode', ['glossary_filter_mode_var'], "
                                  "'strict', str),\n",
                                  "            ('glossary_gender_tracking_bias', "
                                  "['glossary_gender_tracking_bias_var'], 'none', str),\n"
                                  "            ('glossary_enable_relations_tracker', "
                                  "['enable_relations_tracker_checkbox', "
                                  "'glossary_enable_relations_tracker_var'], False, bool),\n"
                                  "            ('glossary_relations_history_mode', "
                                  "['glossary_relations_history_mode_var'], 'previous', str),\n"
                                  "            ('glossary_filter_mode', ['glossary_filter_mode_var'], "
                                  "'strict', str),\n")],
 'src/translator_gui.py': [('        self.glossary_gender_tracking_bias_var = '
                            "self.config.get('glossary_gender_tracking_bias', 'none')\n"
                            '        self.glossary_partial_ratio_gender_only_var = '
                            "self.config.get('glossary_partial_ratio_gender_only', False)\n",
                            '        self.glossary_gender_tracking_bias_var = '
                            "self.config.get('glossary_gender_tracking_bias', 'none')\n"
                            '        self.glossary_enable_relations_tracker_var = '
                            "self.config.get('glossary_enable_relations_tracker', False)\n"
                            '        self.glossary_relations_history_mode_var = '
                            "self.config.get('glossary_relations_history_mode', 'previous')\n"
                            '        self.glossary_partial_ratio_gender_only_var = '
                            "self.config.get('glossary_partial_ratio_gender_only', False)\n"),
                           ("            'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, "
                            "'glossary_gender_tracking_bias_var', "
                            "self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                            "            'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, "
                            "'glossary_partial_ratio_gender_only_var', False) else '0',\n",
                            "            'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, "
                            "'glossary_gender_tracking_bias_var', "
                            "self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                            "            'GLOSSARY_ENABLE_RELATIONS_TRACKER': '1' if getattr(self, "
                            "'glossary_enable_relations_tracker_var', False) else '0',\n"
                            "            'GLOSSARY_RELATIONS_HISTORY_MODE': str(getattr(self, "
                            "'glossary_relations_history_mode_var', "
                            "self.config.get('glossary_relations_history_mode', 'previous'))),\n"
                            "            'GLOSSARY_RELATIONS_FIELD': "
                            "str(self.config.get('glossary_relations_field_name', 'relations')),\n"
                            "            'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, "
                            "'glossary_partial_ratio_gender_only_var', False) else '0',\n"),
                           ("                    'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, "
                            "'glossary_gender_tracking_bias_var', "
                            "self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                            "                    'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, "
                            "'glossary_partial_ratio_gender_only_var', False) else '0',\n",
                            "                    'GLOSSARY_GENDER_TRACKING_BIAS': str(getattr(self, "
                            "'glossary_gender_tracking_bias_var', "
                            "self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                            "                    'GLOSSARY_ENABLE_RELATIONS_TRACKER': '1' if getattr(self, "
                            "'glossary_enable_relations_tracker_var', False) else '0',\n"
                            "                    'GLOSSARY_RELATIONS_HISTORY_MODE': str(getattr(self, "
                            "'glossary_relations_history_mode_var', "
                            "self.config.get('glossary_relations_history_mode', 'previous'))),\n"
                            "                    'GLOSSARY_RELATIONS_FIELD': "
                            "str(self.config.get('glossary_relations_field_name', 'relations')),\n"
                            "                    'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, "
                            "'glossary_partial_ratio_gender_only_var', False) else '0',\n"),
                           ("            ('GLOSSARY_GENDER_TRACKING_BIAS', "
                            "str(self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                            "            ('GLOSSARY_USE_SMART_FILTER', '1' if "
                            "self.config.get('glossary_use_smart_filter', True) else '0'),\n",
                            "            ('GLOSSARY_GENDER_TRACKING_BIAS', "
                            "str(self.config.get('glossary_gender_tracking_bias', 'none'))),\n"
                            "            ('GLOSSARY_ENABLE_RELATIONS_TRACKER', '1' if "
                            "self.config.get('glossary_enable_relations_tracker', False) else '0'),\n"
                            "            ('GLOSSARY_RELATIONS_HISTORY_MODE', "
                            "str(self.config.get('glossary_relations_history_mode', 'previous'))),\n"
                            "            ('GLOSSARY_RELATIONS_FIELD', "
                            "str(self.config.get('glossary_relations_field_name', 'relations'))),\n"
                            "            ('GLOSSARY_USE_SMART_FILTER', '1' if "
                            "self.config.get('glossary_use_smart_filter', True) else '0'),\n"),
                           ("                ('glossary_gender_tracking_bias', "
                            "['glossary_gender_tracking_bias_var'], 'none', str),\n"
                            "                ('glossary_filter_mode', ['glossary_filter_mode_var'], "
                            "'strict', str),\n",
                            "                ('glossary_gender_tracking_bias', "
                            "['glossary_gender_tracking_bias_var'], 'none', str),\n"
                            "                ('glossary_enable_relations_tracker', "
                            "['enable_relations_tracker_checkbox', 'glossary_enable_relations_tracker_var'], "
                            'False, bool),\n'
                            "                ('glossary_relations_history_mode', "
                            "['glossary_relations_history_mode_var'], 'previous', str),\n"
                            "                ('glossary_filter_mode', ['glossary_filter_mode_var'], "
                            "'strict', str),\n")]}


# 2026-11-13 editable placeholder texts (glossary_placeholder_texts / GLOSSARY_PLACEHOLDER_TEXTS): the row
# goes after the GLOSSARY_RELATIONS_FIELD row added above; each pair is (row before + row after) ->
# (before + new row + after), so it applies at every pin and is a no-op where the row already exists.
def _placeholder_texts_rows():
    value = "json.dumps(self.config.get('glossary_placeholder_texts') or {}, ensure_ascii=False)"
    dict_field = "'GLOSSARY_RELATIONS_FIELD': str(self.config.get('glossary_relations_field_name', 'relations')),\n"
    dict_next = "'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, 'glossary_partial_ratio_gender_only_var', False) else '0',\n"
    tuple_field = "('GLOSSARY_RELATIONS_FIELD', str(self.config.get('glossary_relations_field_name', 'relations'))),\n"
    tuple_next = "('GLOSSARY_USE_SMART_FILTER', '1' if self.config.get('glossary_use_smart_filter', True) else '0'),\n"
    pairs = []
    for indent in (" " * 12, " " * 20):
        pairs.append((indent + dict_field + indent + dict_next,
                      indent + dict_field + indent + f"'GLOSSARY_PLACEHOLDER_TEXTS': {value},\n" + indent + dict_next))
    indent = " " * 12
    pairs.append((indent + tuple_field + indent + tuple_next,
                  indent + tuple_field + indent + f"('GLOSSARY_PLACEHOLDER_TEXTS', {value}),\n" + indent + tuple_next))
    return {'src/run_env.py': pairs, 'src/translator_gui.py': pairs}


_PLACEHOLDER_TEXTS_ROWS = _placeholder_texts_rows()


# 2026-11-13 Minimal "Include Relations Column" (glossary_minimal_include_relations /
# GLOSSARY_MINIMAL_INCLUDE_RELATIONS): same (row before + row after) -> (before + new row + after) pairs.
def _minimal_relations_rows():
    placeholder_value = "json.dumps(self.config.get('glossary_placeholder_texts') or {}, ensure_ascii=False)"
    dict_before = f"'GLOSSARY_PLACEHOLDER_TEXTS': {placeholder_value},\n"
    dict_new = ("'GLOSSARY_MINIMAL_INCLUDE_RELATIONS': '1' if getattr(self, "
                "'glossary_minimal_include_relations_var', False) else '0',\n")
    dict_next = "'GLOSSARY_PARTIAL_RATIO_GENDER_ONLY': '1' if getattr(self, 'glossary_partial_ratio_gender_only_var', False) else '0',\n"
    tuple_before = f"('GLOSSARY_PLACEHOLDER_TEXTS', {placeholder_value}),\n"
    tuple_new = ("('GLOSSARY_MINIMAL_INCLUDE_RELATIONS', '1' if "
                 "self.config.get('glossary_minimal_include_relations', False) else '0'),\n")
    tuple_next = "('GLOSSARY_USE_SMART_FILTER', '1' if self.config.get('glossary_use_smart_filter', True) else '0'),\n"
    var_before = "self.glossary_relations_history_mode_var = self.config.get('glossary_relations_history_mode', 'previous')\n"
    var_new = "self.glossary_minimal_include_relations_var = self.config.get('glossary_minimal_include_relations', False)\n"
    var_next = "self.glossary_partial_ratio_gender_only_var = self.config.get('glossary_partial_ratio_gender_only', False)\n"
    map_before = "('include_description', ['include_description_checkbox', 'include_description_var'], False, bool),\n"
    map_new = ("('glossary_minimal_include_relations', ['minimal_include_relations_checkbox', "
               "'glossary_minimal_include_relations_var'], False, bool),\n")
    map_next = "('glossary_use_smart_filter', ['glossary_use_smart_filter_var'], True, bool),\n"

    def pair(indent, before, new, after):
        return (indent + before + indent + after, indent + before + indent + new + indent + after)

    env_rows = [pair(" " * 12, dict_before, dict_new, dict_next), pair(" " * 20, dict_before, dict_new, dict_next),
                pair(" " * 12, tuple_before, tuple_new, tuple_next)]
    var_rows = [pair(" " * 8, var_before, var_new, var_next)]
    map_rows = [pair(" " * 12, map_before, map_new, map_next), pair(" " * 16, map_before, map_new, map_next)]
    return {'src/run_env.py': env_rows, 'src/owner_state.py': var_rows, 'src/settings_persistence.py': map_rows,
            'src/translator_gui.py': env_rows + var_rows + map_rows}


_MINIMAL_RELATIONS_ROWS = _minimal_relations_rows()


# 2026-11-13 a fresh config starts with both Relations Tracker toggles on: the startup config block
# (owner_state._init_config_state, moved from TranslatorGUI.__init__) gets the fresh check before
# "Ensure default values exist" (the pinned sources predate it).
_FRESH_RELATIONS_LINES = (
    "        # A fresh config (no config.json at all, or one Reset Settings to Defaults wrote) starts with the\n"
    "        # Relations Tracker toggles on; an existing config (or a restored / set-aside one) is never touched.\n"
    "        import relations_tracking\n"
    "        if (not self.config and not getattr(self, '_config_recovery_notice', None)\n"
    "                and not os.path.exists(self.config_file_path)):\n"
    "            self.config[relations_tracking.FRESH_DEFAULTS_KEY] = list(relations_tracking.FRESH_DEFAULT_KEYS)\n"
    "        relations_tracking.apply_fresh_defaults(self.config)\n"
    "\n"
)
_FRESH_RELATIONS_PAIR = ("        # Ensure default values exist\n",
                         _FRESH_RELATIONS_LINES + "        # Ensure default values exist\n")
_FRESH_RELATIONS_ROWS = {'src/translator_gui.py': [_FRESH_RELATIONS_PAIR],
                         'src/owner_state.py': [_FRESH_RELATIONS_PAIR]}


# 2026-11-13 search/gemini overhaul (google.com AI Mode with attachments): the "Gemini Free Browser
# Chunking" settings and their env vars are removed. Each row is deleted where a pinned source has it
# (whole lines: the leading newline keeps a 12-space row from matching inside a 16-space one).
_GEMINI_CHUNKING_KEYS = (
    'gemini_free_adaptive_split',
    'gemini_free_html_text_node_transport',
    'gemini_free_subchunk_prompt_chars',
    'gemini_free_subchunk_url_chars',
    'gemini_free_subchunk_safety_chars',
    'gemini_free_min_subchunk_body_chars',
    'gemini_free_subchunk_concurrency',
    'gemini_free_subchunk_start_delay',
    'gemini_free_subchunk_timeout',
    'gemini_free_subchunk_payload_format',
    'gemini_free_html_splitter',
    'gemini_free_subchunk_balancer',
)
_GEMINI_CHUNKING_REMOVED_ROWS = {
    'src/translator_gui.py': (
        "            'GEMINI_FREE_ADAPTIVE_SPLIT': '1' if _bool_config_value('gemini_free_adaptive_split_var', 'gemini_free_adaptive_split', True) else '0',",
        "            'GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT': '1' if _bool_config_value('gemini_free_html_text_node_transport_var', 'gemini_free_html_text_node_transport', True) else '0',",
        "            'GEMINI_FREE_SUBCHUNK_PROMPT_CHARS': _bounded_config_int('gemini_free_subchunk_prompt_chars_var', 'gemini_free_subchunk_prompt_chars', 7000, 300, 7000),",
        "            'GEMINI_FREE_SUBCHUNK_URL_CHARS': _bounded_config_int('gemini_free_subchunk_url_chars_var', 'gemini_free_subchunk_url_chars', 14500, 1000, 200000),",
        "            'GEMINI_FREE_SUBCHUNK_SAFETY_CHARS': _bounded_config_int('gemini_free_subchunk_safety_chars_var', 'gemini_free_subchunk_safety_chars', 600, 0, 20000),",
        "            'GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS': _bounded_config_int('gemini_free_min_subchunk_body_chars_var', 'gemini_free_min_subchunk_body_chars', 80, 1, 50000),",
        "            'GEMINI_FREE_SUBCHUNK_CONCURRENCY': _bounded_config_int('gemini_free_subchunk_concurrency_var', 'gemini_free_subchunk_concurrency', 0, 0, 64),",
        "            'GEMINI_FREE_SUBCHUNK_START_DELAY': _bounded_config_float('gemini_free_subchunk_start_delay_var', 'gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0),",
        "            'GEMINI_FREE_SUBCHUNK_TIMEOUT': _bounded_config_int('gemini_free_subchunk_timeout_var', 'gemini_free_subchunk_timeout', 0, 0, 7200),",
        "            'GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT': _choice_config_value('gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'}),",
        "            'GEMINI_FREE_HTML_SPLITTER': _choice_config_value('gemini_free_html_splitter_var', 'gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'}),",
        "            'GEMINI_FREE_SUBCHUNK_BALANCER': _choice_config_value('gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'}),",
        "                ('gemini_free_adaptive_split', ['gemini_free_adaptive_split_checkbox', 'gemini_free_adaptive_split_var'], True, bool),",
        "                ('gemini_free_html_text_node_transport', ['gemini_free_html_text_node_transport_checkbox', 'gemini_free_html_text_node_transport_var'], True, bool),",
        "                ('gemini_free_subchunk_prompt_chars', ['gemini_free_subchunk_prompt_chars_var'], 7000, lambda v: min(7000, max(300, safe_int(v, 7000)))),",
        "                ('gemini_free_subchunk_url_chars', ['gemini_free_subchunk_url_chars_var'], 14500, lambda v: min(200000, max(1000, safe_int(v, 14500)))),",
        "                ('gemini_free_subchunk_safety_chars', ['gemini_free_subchunk_safety_chars_var'], 600, lambda v: min(20000, max(0, safe_int(v, 600)))),",
        "                ('gemini_free_min_subchunk_body_chars', ['gemini_free_min_subchunk_body_chars_var'], 80, lambda v: min(50000, max(1, safe_int(v, 80)))),",
        "                ('gemini_free_subchunk_concurrency', ['gemini_free_subchunk_concurrency_var'], 0, lambda v: min(64, max(0, safe_int(v, 0)))),",
        "                ('gemini_free_subchunk_start_delay', ['gemini_free_subchunk_start_delay_var'], 5.0, lambda v: min(60.0, max(0.0, safe_float(v, 5.0)))),",
        "                ('gemini_free_subchunk_timeout', ['gemini_free_subchunk_timeout_var'], 0, lambda v: min(7200, max(0, safe_int(v, 0)))),",
        "                ('gemini_free_subchunk_payload_format', ['gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format_combo'], 'auto', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('auto', 'html', 'text') else 'auto'),",
        "                ('gemini_free_html_splitter', ['gemini_free_html_splitter_var', 'gemini_free_html_splitter_combo'], 'beautifulsoup4', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('beautifulsoup4', 'regex') else 'beautifulsoup4'),",
        "                ('gemini_free_subchunk_balancer', ['gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer_combo'], 'balanced', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('balanced', 'greedy') else 'balanced'),",
        "            env_vars_set.append(_update_env('GEMINI_FREE_ADAPTIVE_SPLIT', _config_bool('gemini_free_adaptive_split', True), is_bool=True))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT', _config_bool('gemini_free_html_text_node_transport', True), is_bool=True))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_PROMPT_CHARS', _config_int('gemini_free_subchunk_prompt_chars', 7000, 300, 7000)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_URL_CHARS', _config_int('gemini_free_subchunk_url_chars', 14500, 1000, 200000)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_SAFETY_CHARS', _config_int('gemini_free_subchunk_safety_chars', 600, 0, 20000)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS', _config_int('gemini_free_min_subchunk_body_chars', 80, 1, 50000)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_CONCURRENCY', _config_int('gemini_free_subchunk_concurrency', 0, 0, 64)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_START_DELAY', _config_float('gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_TIMEOUT', _config_int('gemini_free_subchunk_timeout', 0, 0, 7200)))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT', _config_choice('gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'})))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_HTML_SPLITTER', _config_choice('gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'})))",
        "            env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_BALANCER', _config_choice('gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'})))",
        "                ('GEMINI_FREE_ADAPTIVE_SPLIT', '1' if _bool_config('gemini_free_adaptive_split', True) else '0'),",
        "                ('GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT', '1' if _bool_config('gemini_free_html_text_node_transport', True) else '0'),",
        "                ('GEMINI_FREE_SUBCHUNK_PROMPT_CHARS', _int_config('gemini_free_subchunk_prompt_chars', 7000, 300, 7000)),",
        "                ('GEMINI_FREE_SUBCHUNK_URL_CHARS', _int_config('gemini_free_subchunk_url_chars', 14500, 1000, 200000)),",
        "                ('GEMINI_FREE_SUBCHUNK_SAFETY_CHARS', _int_config('gemini_free_subchunk_safety_chars', 600, 0, 20000)),",
        "                ('GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS', _int_config('gemini_free_min_subchunk_body_chars', 80, 1, 50000)),",
        "                ('GEMINI_FREE_SUBCHUNK_CONCURRENCY', _int_config('gemini_free_subchunk_concurrency', 0, 0, 64)),",
        "                ('GEMINI_FREE_SUBCHUNK_START_DELAY', _float_config('gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0)),",
        "                ('GEMINI_FREE_SUBCHUNK_TIMEOUT', _int_config('gemini_free_subchunk_timeout', 0, 0, 7200)),",
        "                ('GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT', _choice_config('gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'})),",
        "                ('GEMINI_FREE_HTML_SPLITTER', _choice_config('gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'})),",
        "                ('GEMINI_FREE_SUBCHUNK_BALANCER', _choice_config('gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'})),",
    ),
    'src/settings_persistence.py': (
        "            ('gemini_free_adaptive_split', ['gemini_free_adaptive_split_checkbox', 'gemini_free_adaptive_split_var'], True, bool),",
        "            ('gemini_free_html_text_node_transport', ['gemini_free_html_text_node_transport_checkbox', 'gemini_free_html_text_node_transport_var'], True, bool),",
        "            ('gemini_free_subchunk_prompt_chars', ['gemini_free_subchunk_prompt_chars_var'], 7000, lambda v: min(7000, max(300, safe_int(v, 7000)))),",
        "            ('gemini_free_subchunk_url_chars', ['gemini_free_subchunk_url_chars_var'], 14500, lambda v: min(200000, max(1000, safe_int(v, 14500)))),",
        "            ('gemini_free_subchunk_safety_chars', ['gemini_free_subchunk_safety_chars_var'], 600, lambda v: min(20000, max(0, safe_int(v, 600)))),",
        "            ('gemini_free_min_subchunk_body_chars', ['gemini_free_min_subchunk_body_chars_var'], 80, lambda v: min(50000, max(1, safe_int(v, 80)))),",
        "            ('gemini_free_subchunk_concurrency', ['gemini_free_subchunk_concurrency_var'], 0, lambda v: min(64, max(0, safe_int(v, 0)))),",
        "            ('gemini_free_subchunk_start_delay', ['gemini_free_subchunk_start_delay_var'], 5.0, lambda v: min(60.0, max(0.0, safe_float(v, 5.0)))),",
        "            ('gemini_free_subchunk_timeout', ['gemini_free_subchunk_timeout_var'], 0, lambda v: min(7200, max(0, safe_int(v, 0)))),",
        "            ('gemini_free_subchunk_payload_format', ['gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format_combo'], 'auto', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('auto', 'html', 'text') else 'auto'),",
        "            ('gemini_free_html_splitter', ['gemini_free_html_splitter_var', 'gemini_free_html_splitter_combo'], 'beautifulsoup4', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('beautifulsoup4', 'regex') else 'beautifulsoup4'),",
        "            ('gemini_free_subchunk_balancer', ['gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer_combo'], 'balanced', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('balanced', 'greedy') else 'balanced'),",
        "        env_vars_set.append(_update_env('GEMINI_FREE_ADAPTIVE_SPLIT', _config_bool('gemini_free_adaptive_split', True), is_bool=True))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT', _config_bool('gemini_free_html_text_node_transport', True), is_bool=True))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_PROMPT_CHARS', _config_int('gemini_free_subchunk_prompt_chars', 7000, 300, 7000)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_URL_CHARS', _config_int('gemini_free_subchunk_url_chars', 14500, 1000, 200000)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_SAFETY_CHARS', _config_int('gemini_free_subchunk_safety_chars', 600, 0, 20000)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS', _config_int('gemini_free_min_subchunk_body_chars', 80, 1, 50000)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_CONCURRENCY', _config_int('gemini_free_subchunk_concurrency', 0, 0, 64)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_START_DELAY', _config_float('gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_TIMEOUT', _config_int('gemini_free_subchunk_timeout', 0, 0, 7200)))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT', _config_choice('gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'})))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_HTML_SPLITTER', _config_choice('gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'})))",
        "        env_vars_set.append(_update_env('GEMINI_FREE_SUBCHUNK_BALANCER', _config_choice('gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'})))",
    ),
    'src/run_env.py': (
        "            'GEMINI_FREE_ADAPTIVE_SPLIT': '1' if _bool_config_value('gemini_free_adaptive_split_var', 'gemini_free_adaptive_split', True) else '0',",
        "            'GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT': '1' if _bool_config_value('gemini_free_html_text_node_transport_var', 'gemini_free_html_text_node_transport', True) else '0',",
        "            'GEMINI_FREE_SUBCHUNK_PROMPT_CHARS': _bounded_config_int('gemini_free_subchunk_prompt_chars_var', 'gemini_free_subchunk_prompt_chars', 7000, 300, 7000),",
        "            'GEMINI_FREE_SUBCHUNK_URL_CHARS': _bounded_config_int('gemini_free_subchunk_url_chars_var', 'gemini_free_subchunk_url_chars', 14500, 1000, 200000),",
        "            'GEMINI_FREE_SUBCHUNK_SAFETY_CHARS': _bounded_config_int('gemini_free_subchunk_safety_chars_var', 'gemini_free_subchunk_safety_chars', 600, 0, 20000),",
        "            'GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS': _bounded_config_int('gemini_free_min_subchunk_body_chars_var', 'gemini_free_min_subchunk_body_chars', 80, 1, 50000),",
        "            'GEMINI_FREE_SUBCHUNK_CONCURRENCY': _bounded_config_int('gemini_free_subchunk_concurrency_var', 'gemini_free_subchunk_concurrency', 0, 0, 64),",
        "            'GEMINI_FREE_SUBCHUNK_START_DELAY': _bounded_config_float('gemini_free_subchunk_start_delay_var', 'gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0),",
        "            'GEMINI_FREE_SUBCHUNK_TIMEOUT': _bounded_config_int('gemini_free_subchunk_timeout_var', 'gemini_free_subchunk_timeout', 0, 0, 7200),",
        "            'GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT': _choice_config_value('gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'}),",
        "            'GEMINI_FREE_HTML_SPLITTER': _choice_config_value('gemini_free_html_splitter_var', 'gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'}),",
        "            'GEMINI_FREE_SUBCHUNK_BALANCER': _choice_config_value('gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'}),",
        "                ('GEMINI_FREE_ADAPTIVE_SPLIT', '1' if _bool_config('gemini_free_adaptive_split', True) else '0'),",
        "                ('GEMINI_FREE_HTML_TEXT_NODE_TRANSPORT', '1' if _bool_config('gemini_free_html_text_node_transport', True) else '0'),",
        "                ('GEMINI_FREE_SUBCHUNK_PROMPT_CHARS', _int_config('gemini_free_subchunk_prompt_chars', 7000, 300, 7000)),",
        "                ('GEMINI_FREE_SUBCHUNK_URL_CHARS', _int_config('gemini_free_subchunk_url_chars', 14500, 1000, 200000)),",
        "                ('GEMINI_FREE_SUBCHUNK_SAFETY_CHARS', _int_config('gemini_free_subchunk_safety_chars', 600, 0, 20000)),",
        "                ('GEMINI_FREE_MIN_SUBCHUNK_BODY_CHARS', _int_config('gemini_free_min_subchunk_body_chars', 80, 1, 50000)),",
        "                ('GEMINI_FREE_SUBCHUNK_CONCURRENCY', _int_config('gemini_free_subchunk_concurrency', 0, 0, 64)),",
        "                ('GEMINI_FREE_SUBCHUNK_START_DELAY', _float_config('gemini_free_subchunk_start_delay', 5.0, 0.0, 60.0)),",
        "                ('GEMINI_FREE_SUBCHUNK_TIMEOUT', _int_config('gemini_free_subchunk_timeout', 0, 0, 7200)),",
        "                ('GEMINI_FREE_SUBCHUNK_PAYLOAD_FORMAT', _choice_config('gemini_free_subchunk_payload_format', 'auto', {'auto', 'html', 'text'})),",
        "                ('GEMINI_FREE_HTML_SPLITTER', _choice_config('gemini_free_html_splitter', 'beautifulsoup4', {'beautifulsoup4', 'regex'})),",
        "                ('GEMINI_FREE_SUBCHUNK_BALANCER', _choice_config('gemini_free_subchunk_balancer', 'balanced', {'balanced', 'greedy'})),",
    ),
    'src/mobile/tools/frozen_desktop_tables.py': (
        "            ('gemini_free_adaptive_split', ['gemini_free_adaptive_split_checkbox', 'gemini_free_adaptive_split_var'], True, bool),",
        "            ('gemini_free_html_text_node_transport', ['gemini_free_html_text_node_transport_checkbox', 'gemini_free_html_text_node_transport_var'], True, bool),",
        "            ('gemini_free_subchunk_prompt_chars', ['gemini_free_subchunk_prompt_chars_var'], 7000, lambda v: min(7000, max(300, safe_int(v, 7000)))),",
        "            ('gemini_free_subchunk_url_chars', ['gemini_free_subchunk_url_chars_var'], 14500, lambda v: min(200000, max(1000, safe_int(v, 14500)))),",
        "            ('gemini_free_subchunk_safety_chars', ['gemini_free_subchunk_safety_chars_var'], 600, lambda v: min(20000, max(0, safe_int(v, 600)))),",
        "            ('gemini_free_min_subchunk_body_chars', ['gemini_free_min_subchunk_body_chars_var'], 80, lambda v: min(50000, max(1, safe_int(v, 80)))),",
        "            ('gemini_free_subchunk_concurrency', ['gemini_free_subchunk_concurrency_var'], 0, lambda v: min(64, max(0, safe_int(v, 0)))),",
        "            ('gemini_free_subchunk_start_delay', ['gemini_free_subchunk_start_delay_var'], 5.0, lambda v: min(60.0, max(0.0, safe_float(v, 5.0)))),",
        "            ('gemini_free_subchunk_timeout', ['gemini_free_subchunk_timeout_var'], 0, lambda v: min(7200, max(0, safe_int(v, 0)))),",
        "            ('gemini_free_subchunk_payload_format', ['gemini_free_subchunk_payload_format_var', 'gemini_free_subchunk_payload_format_combo'], 'auto', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('auto', 'html', 'text') else 'auto'),",
        "            ('gemini_free_html_splitter', ['gemini_free_html_splitter_var', 'gemini_free_html_splitter_combo'], 'beautifulsoup4', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('beautifulsoup4', 'regex') else 'beautifulsoup4'),",
        "            ('gemini_free_subchunk_balancer', ['gemini_free_subchunk_balancer_var', 'gemini_free_subchunk_balancer_combo'], 'balanced', lambda v: str(v).strip().lower() if str(v).strip().lower() in ('balanced', 'greedy') else 'balanced'),",
    ),
}


def _removed_rows(rows):
    return [("\n" + row + "\n", "\n") for row in rows]

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
        "what": "New setting 'Fix Missing Tag Bracket' (fix_missing_tag_bracket, on) restores the > an LLM dropped "
                "in <p\"Hello\"</p> before parsing; new QA check (check_missing_tag_bracket, on) reports it as an "
                "LLM token issue, repaired by Resolve QA",
        "source": {
            **_MISSING_BRACKET_LEGACY,
            **_MISSING_BRACKET_SCHEMA,
            **_MISSING_BRACKET_QA,
        },
        "values": {},
    },
    {
        "date": "2026-11-01",
        "what": "Glossarion Mobile only: a QA scan's workers are the setting up to the device's cores, half the "
                "cores when none is set (the cap of 2 is gone); the functions run only where process pools are "
                "unavailable, the desktop never calls them",
        "source": {
            "src/qa_scan_runtime.py": [(
                "#: Glossarion Mobile runs QA scans on a thread pool (Android / iOS have no process pools) with at\n"
                "#: most this many workers (``AI_HUNTER_MAX_WORKERS``). The desktop never applies either.\n"
                "MOBILE_QA_MAX_WORKERS = 2\n"
                "\n"
                "\n"
                "def mobile_qa_forcing_active():\n",
                "# Glossarion Mobile runs QA scans on a thread pool (Android / iOS have no process pools) with\n"
                "# ``mobile_qa_max_workers`` workers (``AI_HUNTER_MAX_WORKERS``). The desktop never applies either.\n"
                "def mobile_qa_forcing_active():\n",
            ), (
                "def mobile_qa_max_workers():\n"
                '    """Worker cap of a mobile scan: a smaller positive ``AI_HUNTER_MAX_WORKERS`` wins."""\n'
                "    try:\n"
                '        configured = int(str(os.getenv("AI_HUNTER_MAX_WORKERS", "") or "").strip() or 0)\n'
                "    except (TypeError, ValueError):\n"
                "        configured = 0\n"
                "    if 0 < configured < MOBILE_QA_MAX_WORKERS:\n"
                "        return configured\n"
                "    return MOBILE_QA_MAX_WORKERS\n",
                "def mobile_qa_max_workers(configured=None):\n"
                '    """Workers of a mobile scan: the setting (``AI_HUNTER_MAX_WORKERS`` unless given) up to the device\'s\n'
                '    cores. No setting, or the desktop\'s 0 ("all cores"), means half the cores."""\n'
                "    if configured is None:\n"
                '        configured = os.getenv("AI_HUNTER_MAX_WORKERS", "")\n'
                "    cores = os.cpu_count() or 4\n"
                "    try:\n"
                '        configured = int(str(configured or "").strip() or 0)\n'
                "    except (TypeError, ValueError):\n"
                "        configured = 0\n"
                "    if configured > 0:\n"
                "        return min(configured, cores)\n"
                "    return max(1, cores // 2)\n",
            )],
        },
        "values": {},
    },
    {
        "date": "2026-11-02",
        "what": "An OPF manifest href is a URL: 'part%20001.xhtml' is the file 'part 001.xhtml', chapter 1. "
                "It is decoded where it is read",
        "source": {
            "src/scan_html_folder.py": [
                ("import os\nimport hashlib\n", "import os\nfrom urllib.parse import unquote\nimport hashlib\n"),
                ("href = item.get('href')", "href = unquote(item.get('href') or '')"),
                ('item.get("id"): os.path.basename(item.get("href") or "")',
                 'item.get("id"): os.path.basename(unquote(item.get("href") or ""))'),
            ],
            "src/async_api_processor.py": [
                ("import os\nimport sys\nimport re\n", "import os\nfrom urllib.parse import unquote\nimport sys\nimport re\n"),
                ('href = item.get("href")', 'href = unquote(item.get("href") or "")'),
            ],
        },
        "values": {},
    },
    {
        "date": "2026-11-04",
        "what": "Multi-instance translation: the QA scanner reads the user's selection through instance_runs.user_selection (every file of a running multi-instance run), its live config under a "
                "key scope is that instance's (instance_plan.config_for_instance), and the AI truncation key pool keeps only the scanning instance's keys (off for it when it has none)",
        "source": {
            "src/QA_Scanner_GUI.py": [
                (
                    '        current_epub_files = []\n'
                    "        if hasattr(self, 'selected_files') and self.selected_files:\n"
                    '            current_epub_files = [\n'
                    '                f for f in self.selected_files\n'
                    "                if f.lower().endswith(('.epub', '.txt', '.pdf', '.md', '.html', '.htm', '.xhtml'))\n",
                    '        current_epub_files = []\n'
                    '        from instance_runs import user_selection  # every file of a multi-instance run\n'
                    '\n'
                    '        selected_files = user_selection(self)\n'
                    '        if selected_files:\n'
                    '            current_epub_files = [\n'
                    '                f for f in selected_files\n'
                    "                if f.lower().endswith(('.epub', '.txt', '.pdf', '.md', '.html', '.htm', '.xhtml'))\n",
                ),
                (
                    '                files = []\n'
                    '                try:\n'
                    '                    files = [\n'
                    "                        f for f in (getattr(self, 'selected_files', []) or [])\n"
                    '                        if isinstance(f, str) and f.lower().endswith(source_exts)\n',
                    '                files = []\n'
                    '                try:\n'
                    '                    from instance_runs import user_selection  # every file of a multi-instance run\n'
                    '\n'
                    '                    files = [\n'
                    '                        f for f in user_selection(self)\n'
                    '                        if isinstance(f, str) and f.lower().endswith(source_exts)\n',
                ),
            ],
            "src/qa_scan_runtime.py": [
                (
                    '    cfg.setdefault("rotation_frequency", cfg.get("rotation_frequency", 1))\n'
                    '    return cfg, api_key, model\n',
                    '    cfg.setdefault("rotation_frequency", cfg.get("rotation_frequency", 1))\n'
                    '    # Multi-instance: under a key scope the scan runs as that instance (its keys in every pool; an\n'
                    "    # instance >= 2 on its own key in place of the main GUI key, which is instance 1's)\n"
                    '    import instance_plan\n'
                    '\n'
                    '    scope = instance_plan.key_scope()\n'
                    '    if scope is not None:\n'
                    '        try:\n'
                    '            cfg = instance_plan.config_for_instance(cfg, scope)\n'
                    '            api_key, model = cfg.get("api_key", api_key), cfg.get("model", model)\n'
                    '        except ValueError:\n'
                    "            pass  # no Translation key of its own: the API client's key scope still applies\n"
                    '    return cfg, api_key, model\n',
                ),
            ],
            "src/scan_html_folder.py": [
                (
                    '            if not isinstance(_ai_pool_keys, list):\n'
                    '                _ai_pool_keys = []\n'
                    '            _ai_pool_usable_keys = [\n',
                    '            if not isinstance(_ai_pool_keys, list):\n'
                    '                _ai_pool_keys = []\n'
                    "            # Multi-instance: only the scanning instance's keys; with none of them the pool is off\n"
                    '            # for it (its Translation keys do the work)\n'
                    '            from instance_plan import scope_keys\n'
                    '            _ai_pool_scoped = scope_keys(_ai_pool_keys)\n'
                    '            if _ai_pool_scoped is not _ai_pool_keys:\n'
                    '                _ai_pool_keys = _ai_pool_scoped\n'
                    '                _ai_pool_enabled = _ai_pool_enabled and any(\n'
                    "                    isinstance(key, dict) and key.get('enabled', True) and key.get('model')\n"
                    '                    for key in _ai_pool_keys\n'
                    '                )\n'
                    '            _ai_pool_usable_keys = [\n',
                ),
            ],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga renderer: auto font size respects bubble and box boundaries (the bubble interior is read from the page and every line is fitted to its width at that line's height, outline and shadow included)",
        "source": {
            "src/manga_translator.py": [
                (
                    '                        # Last resort: use simple bounding box\n'
                    '                        render_x, render_y, render_w, render_h = x, y, w, h\n'
                    '                    # Cache the result\n'
                    '                    region._cached_safe_area = (render_x, render_y, render_w, render_h)\n'
                    ,
                    '                        # Last resort: use simple bounding box\n'
                    '                        render_x, render_y, render_w, render_h = x, y, w, h\n'
                    '                    # Keep text inside the bubble / box actually drawn on the page\n'
                    '                    render_x, render_y, render_w, render_h = self._container_render_area(\n'
                    '                        image, region, (render_x, render_y, render_w, render_h), regions)\n'
                    '                    # Cache the result\n'
                    '                    region._cached_safe_area = (render_x, render_y, render_w, render_h)\n'
                    ,
                ),
                (
                    '     \n'
                    '    def _pil_word_wrap(self, text: str, font_path: str, roi_width: int, roi_height: int,\n'
                    '                       init_font_size: int, min_font_size: int, draw: ImageDraw) -> Tuple[str, int]:\n'
                    '        """Binary search for perfect-fit font sizing.\n'
                    '        \n'
                    ,
                    '     \n'
                    '    def _pil_word_wrap(self, text: str, font_path: str, roi_width: int, roi_height: int,\n'
                    '                       init_font_size: int, min_font_size: int, draw: ImageDraw,\n'
                    '                       container: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:\n'
                    '        """Binary search for perfect-fit font sizing.\n'
                    '        \n'
                    ,
                ),
                (
                    '        When strict_text_wrapping is enabled:\n'
                    '        - Falls back to original column-based algorithm (legacy behavior)\n'
                    '        """\n'
                    '        from hyphen_textwrap import wrap as hyphen_wrap\n'
                    '        import statistics\n'
                    ,
                    '        When strict_text_wrapping is enabled:\n'
                    '        - Falls back to original column-based algorithm (legacy behavior)\n'
                    '\n'
                    '        With a ``container`` (bubble interior from _find_text_container) every line is fitted to\n'
                    "        the interior's width at its own height instead (_fit_lines_to_container).\n"
                    '        """\n'
                    '        if container is not None:\n'
                    '            wrapped, size, _fits = self._fit_lines_to_container(\n'
                    '                text, font_path, container, init_font_size, min_font_size, draw)\n'
                    '            return wrapped, size\n'
                    '        from hyphen_textwrap import wrap as hyphen_wrap\n'
                    '        import statistics\n'
                    ,
                ),
                (
                    '        return safe_x, safe_y, safe_width, safe_height\n'
                    '    \n'
                    '    def _fit_text_to_region(self, text: str, max_width: int, max_height: int, draw: ImageDraw, region: TextRegion = None, use_as_is: bool = False) -> Tuple[int, List[str]]:\n'
                    '        """Find optimal font size using comic-translate\'s pil_word_wrap algorithm with algorithm-aware adjustments\n'
                    ,
                    '        return safe_x, safe_y, safe_width, safe_height\n'
                    '    \n'
                    '    def _text_ink_pads(self, font_size: int) -> Tuple[int, int, int, int]:\n'
                    '        """Pixels the outline stroke and the drop shadow add around a line\'s glyph box.\n'
                    '\n'
                    "        Returns (left, top, right, bottom). Mirrors render_translated_text's\n"
                    "        ow = max(1, font_size // outline_width_factor) and _draw_text_shadow's offset and blur.\n"
                    '        """\n'
                    '        try:\n'
                    "            ow = max(1, int(font_size) // max(1, int(getattr(self, 'outline_width_factor', 15) or 15)))\n"
                    '        except Exception:\n'
                    '            ow = 1\n'
                    '        left = top = right = bottom = ow\n'
                    "        if getattr(self, 'shadow_enabled', False):\n"
                    '            try:\n'
                    "                blur = max(0, int(getattr(self, 'shadow_blur', 0) or 0))\n"
                    "                sx = int(getattr(self, 'shadow_offset_x', 0) or 0)\n"
                    "                sy = int(getattr(self, 'shadow_offset_y', 0) or 0)\n"
                    '            except Exception:\n'
                    '                blur = sx = sy = 0\n'
                    '            left, right = max(left, blur - sx), max(right, blur + sx)\n'
                    '            top, bottom = max(top, blur - sy), max(bottom, blur + sy)\n'
                    '        return left, top, right, bottom\n'
                    '\n'
                    '    def _find_text_container(self, image: np.ndarray, area, clip_bounds=None) -> Optional[Dict[str, Any]]:\n'
                    '        """Interior of the speech bubble or narration box a region\'s text sits in, read from the page.\n'
                    '\n'
                    "        The (cleaned) page is flood-filled from the middle of ``area`` with the bubble's own fill\n"
                    "        colour, inside ``area`` united with ``clip_bounds`` (the detector's bubble box). Holes left\n"
                    '        by leftover text are filled, and the interior is kept a margin away from the bubble outline.\n'
                    '        Returns None when the middle of the area is not a uniform fill (free text over art).\n'
                    '\n'
                    '        Returned dict: ``rect`` (x, y, w, h) to lay text out in (its centre column is the\n'
                    "        interior's centre), ``cx``, ``row0`` and per-row ``left`` / ``right`` (pixels of interior\n"
                    '        left/right of ``cx`` on row ``row0 + i``, -1 where ``cx`` is outside it), ``margin``.\n'
                    '        """\n'
                    "        if image is None or not hasattr(image, 'shape') or len(image.shape) < 2:\n"
                    '            return None\n'
                    '        ih, iw = image.shape[:2]\n'
                    '        ax, ay, aw, ah = (int(round(float(v))) for v in area)\n'
                    '        if aw < 8 or ah < 8:\n'
                    '            return None\n'
                    '        x0, y0, x1, y1 = ax, ay, ax + aw, ay + ah\n'
                    '        if clip_bounds:\n'
                    '            try:\n'
                    '                bx, by, bw, bh = (int(round(float(v))) for v in clip_bounds)\n'
                    '                if bw > 0 and bh > 0:\n'
                    '                    x0, y0 = min(x0, bx), min(y0, by)\n'
                    '                    x1, y1 = max(x1, bx + bw), max(y1, by + bh)\n'
                    '            except Exception:\n'
                    '                pass\n'
                    '        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(iw, x1), min(ih, y1)\n'
                    '        if x1 - x0 < 8 or y1 - y0 < 8:\n'
                    '            return None\n'
                    '        win = image[y0:y1, x0:x1]\n'
                    '        if win.ndim == 2:\n'
                    '            win = win[:, :, None]\n'
                    '        win = win[:, :, :3].astype(np.int16)\n'
                    '        # Seed: the middle half of the text area, where the cleaned bubble fill is\n'
                    '        sx0, sx1 = max(ax + aw // 4, x0) - x0, min(ax + aw - aw // 4, x1) - x0\n'
                    '        sy0, sy1 = max(ay + ah // 4, y0) - y0, min(ay + ah - ah // 4, y1) - y0\n'
                    '        if sx1 <= sx0 or sy1 <= sy0:\n'
                    '            return None\n'
                    '        seed = np.median(win[sy0:sy1, sx0:sx1].reshape(-1, win.shape[2]), axis=0)\n'
                    '        near = (np.abs(win - seed).max(axis=2) <= 32).astype(np.uint8)\n'
                    '        if float(near[sy0:sy1, sx0:sx1].mean()) < 0.55:\n'
                    '            return None  # busy background: free text, no container\n'
                    '        near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # cut 1-2 px leaks\n'
                    '        count, labels = cv2.connectedComponents(near, connectivity=4)\n'
                    '        if count <= 1:\n'
                    '            return None\n'
                    '        hits = np.bincount(labels[sy0:sy1, sx0:sx1].ravel(), minlength=count)\n'
                    '        hits[0] = 0\n'
                    '        label = int(hits.argmax())\n'
                    '        if hits[label] == 0:\n'
                    '            return None\n'
                    '        comp = (labels == label).astype(np.uint8)\n'
                    '        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n'
                    '        filled = np.zeros_like(comp)\n'
                    '        cv2.drawContours(filled, contours, -1, 1, thickness=-1)  # fill holes (leftover text, specks)\n'
                    "        # A bubble's interior covers the middle of its text area; a patch between speed lines or\n"
                    '        # other art strokes does not (free text)\n'
                    '        if int(filled.sum()) < 0.3 * filled.size or float(filled[sy0:sy1, sx0:sx1].mean()) < 0.9:\n'
                    '            return None\n'
                    '        ys, xs = np.nonzero(filled)\n'
                    '        margin = int(max(3, min(16, round(0.06 * min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))))\n'
                    '        # Margin from the drawn outline only: window edges are not outlines (border value 1)\n'
                    '        to_outline = cv2.distanceTransform(\n'
                    '            cv2.copyMakeBorder(filled, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1), cv2.DIST_L2, 5)[1:-1, 1:-1]\n'
                    '        to_wall = cv2.distanceTransform(\n'
                    '            cv2.copyMakeBorder(filled, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0), cv2.DIST_L2, 5)[1:-1, 1:-1]\n'
                    '        allowed = (to_outline > margin) & (filled > 0)\n'
                    '        depth = np.where(allowed, to_wall, 0)\n'
                    '        dmax = float(depth.max())\n'
                    '        if dmax <= 0:\n'
                    '            return None\n'
                    '        # Centre: middle of the deepest ridge (one point for an oval, a line for a wide box)\n'
                    '        ridge_y, ridge_x = np.nonzero(depth >= dmax * 0.9)\n'
                    '        cx, cy = int(round(ridge_x.mean())), int(round(ridge_y.mean()))\n'
                    '        if not allowed[cy, cx]:\n'
                    '            cy, cx = (int(v) for v in np.unravel_index(int(depth.argmax()), depth.shape))\n'
                    '\n'
                    '        def _leading_true(a):\n'
                    '            return np.where(a.all(axis=1), a.shape[1], a.argmin(axis=1))\n'
                    '\n'
                    '        left = (_leading_true(allowed[:, cx::-1]) - 1).astype(np.int32)\n'
                    '        right = (_leading_true(allowed[:, cx:]) - 1).astype(np.int32)\n'
                    '        up, down = allowed[cy::-1, cx], allowed[cy:, cx]\n'
                    '        top = cy - (len(up) if up.all() else int(up.argmin())) + 1\n'
                    '        bottom = cy + (len(down) if down.all() else int(down.argmin())) - 1\n'
                    '        half = int(np.minimum(left, right)[top:bottom + 1].max())\n'
                    '        if half < 4 or bottom - top < 8:\n'
                    '            return None\n'
                    '        return {\n'
                    "            'rect': (x0 + cx - half, y0 + top, 2 * half + 1, bottom - top + 1),\n"
                    "            'cx': x0 + cx, 'row0': y0, 'left': left, 'right': right, 'margin': margin,\n"
                    '        }\n'
                    '\n'
                    '    def _container_render_area(self, image: np.ndarray, region: TextRegion, area,\n'
                    '                               source_regions=None) -> Tuple[int, int, int, int]:\n'
                    '        """Render area for auto-sized text that respects the bubble / box drawn on the page.\n'
                    '\n'
                    '        With a container (see _find_text_container) the area becomes its interior, and the\n'
                    '        container is stored on ``region._fit_container`` for _fit_text_to_region. Without one\n'
                    '        (free text over art) the area only loses what the outline stroke and shadow add.\n'
                    '        ``source_regions`` are the regions render_translated_text was given: with several,\n'
                    "        it lays out _adjust_overlapping_regions' copies, which carry no bubble_bounds.\n"
                    '        """\n'
                    '        region._fit_container = None\n'
                    "        if not getattr(self, 'constrain_to_bubble', True):\n"
                    '            return area\n'
                    '        # A safe area the user switched on is an explicit limit: stay inside it\n'
                    "        clip = None if getattr(self, 'safe_area_enabled', False) else getattr(region, 'bubble_bounds', None)\n"
                    "        idx = getattr(region, 'original_index', None)\n"
                    "        if (clip is None and not getattr(self, 'safe_area_enabled', False) and source_regions\n"
                    '                and isinstance(idx, int) and 0 <= idx < len(source_regions)\n'
                    "                and tuple(region.bounding_box) == tuple(getattr(region, 'original_bbox', None) or ())):\n"
                    "            # the copy's own region, unless the overlap adjustment moved the copy off its place\n"
                    "            clip = getattr(source_regions[idx], 'bubble_bounds', None)\n"
                    '        try:\n'
                    '            container = self._find_text_container(image, area, clip)\n'
                    '        except Exception as e:\n'
                    '            container = None\n'
                    '            self._log(f"  ⚠️ Bubble interior detection failed: {e}", "debug")\n'
                    '        if container is not None:\n'
                    '            region._fit_container = container\n'
                    "            return container['rect']\n"
                    '        x, y, w, h = area\n'
                    '        try:\n'
                    '            l, t, r, b = self._text_ink_pads(int(self.max_font_size_limit))\n'
                    '        except Exception:\n'
                    '            return area\n'
                    '        if w - l - r >= 16 and h - t - b >= 16:\n'
                    '            return x + l, y + t, w - l - r, h - t - b\n'
                    '        return area\n'
                    '\n'
                    '    def _fit_lines_to_container(self, text: str, font_path: str, container: Dict[str, Any],\n'
                    '                                init_font_size: int, min_font_size: int, draw: ImageDraw) -> Tuple[str, int, bool]:\n'
                    '        """Largest font size whose wrapped lines all stay inside a bubble interior.\n'
                    '\n'
                    '        Lays the block out the way render_translated_text does (vertically centred in\n'
                    "        ``container['rect']``, every line centred on its centre column) and checks each line's ink,\n"
                    "        outline stroke and shadow included, against the interior's width on the rows that line\n"
                    '        covers, so lines get shorter where an oval narrows. Returns (wrapped_text, size, fits).\n'
                    '        """\n'
                    '        import math\n'
                    '        words = text.split()\n'
                    '        min_font_size = int(min_font_size)\n'
                    '        init_font_size = max(int(init_font_size), min_font_size)\n'
                    '        if not words:\n'
                    '            return text, min_font_size, True\n'
                    "        rx, ry, rw, rh = container['rect']\n"
                    "        cx, row0 = container['cx'], container['row0']\n"
                    "        left, right = container['left'], container['right']\n"
                    '        nrows = len(left)\n'
                    '        try:\n'
                    '            mixed = not self._is_truly_custom_font()  # the renderer mixes in the emote font then\n'
                    '        except Exception:\n'
                    '            mixed = False\n'
                    '\n'
                    '        def make(size):\n'
                    '            try:\n'
                    '                font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()\n'
                    '            except Exception:\n'
                    '                font = ImageFont.load_default()\n'
                    '            emote = self._get_emote_fallback_font(size) if mixed else None\n'
                    '            pads = self._text_ink_pads(size)\n'
                    '            word_cache = {}\n'
                    "            space_w = self._line_width_emote_mixed(draw, ' ', font, emote) if emote is not None else 0\n"
                    '\n'
                    '            def measure(ws):\n'
                    '                """(left bearing, ink top, ink bottom, render width) of a line."""\n'
                    '                if emote is None:\n'
                    "                    bb = draw.textbbox((0, 0), ' '.join(ws), font=font)\n"
                    '                    return bb[0], bb[1], bb[3], bb[2] - bb[0]\n'
                    '                parts = []\n'
                    '                for wd in ws:\n'
                    '                    part = word_cache.get(wd)\n'
                    '                    if part is None:\n'
                    '                        part = word_cache[wd] = (draw.textbbox((0, 0), wd, font=font),\n'
                    '                                                 self._line_width_emote_mixed(draw, wd, font, emote))\n'
                    '                    parts.append(part)\n'
                    '                return (parts[0][0][0], min(p[0][1] for p in parts), max(p[0][3] for p in parts),\n'
                    '                        sum(p[1] for p in parts) + space_w * (len(ws) - 1))\n'
                    '\n'
                    '            def line_height(first):\n'
                    '                bb = draw.textbbox((0, 0), \' \'.join(first) if first else "Ay", font=font)\n'
                    '                return max(size * 1.2, (bb[3] - bb[1]) * 1.1)\n'
                    '\n'
                    '            def fits_at(ws, ty, cap=None):\n'
                    '                bb0, top, bottom, width = measure(ws)\n'
                    '                if cap is not None and width > cap:\n'
                    '                    return False\n'
                    '                a = int(math.floor(ty + top - pads[1])) - row0\n'
                    '                b = int(math.ceil(ty + bottom + pads[3])) - row0\n'
                    '                if a < 0 or b > nrows or b <= a:\n'
                    '                    return False\n'
                    '                room_l, room_r = int(left[a:b].min()), int(right[a:b].min())\n'
                    '                if room_l < 0 or room_r < 0:\n'
                    '                    return False\n'
                    '                tx = rx + (rw - width) // 2\n'
                    '                return (cx - (tx + min(0, bb0) - pads[0]) <= room_l\n'
                    '                        and (tx + width + max(0, bb0) + pads[2] - 1) - cx <= room_r)\n'
                    '\n'
                    '            def wrap(n, lh, cap=None):\n'
                    '                total = n * lh\n'
                    '                start = max(ry, min(ry + (rh - total) // 2, ry + rh - total))\n'
                    '                out, i = [], 0\n'
                    '                for k in range(n):\n'
                    '                    if i >= len(words):\n'
                    '                        return None\n'
                    '                    ty = start + k * lh\n'
                    '                    cur = [words[i]]\n'
                    '                    if not fits_at(cur, ty, cap):\n'
                    '                        return None\n'
                    '                    i += 1\n'
                    '                    # Leave a word for every line still to come: n lines are laid out at the\n'
                    '                    # n-line positions, so running out of words early is not an (n-1)-line fit\n'
                    '                    while (i < len(words) - (n - 1 - k)\n'
                    '                           and fits_at(cur + [words[i]], ty, cap)):\n'
                    '                        cur.append(words[i])\n'
                    '                        i += 1\n'
                    '                    out.append(cur)\n'
                    '                return out if i >= len(words) else None\n'
                    '\n'
                    '            return measure, line_height, wrap, pads\n'
                    '\n'
                    '        def orphaned(lines):\n'
                    '            """A line holding one 1-2 letter word ("I", "A", "IT.") that the other lines left alone."""\n'
                    '            return len(lines) > 1 and any(\n'
                    '                len(ws) == 1 and len(ws[0].strip(\'.,!?~;:"\\\'\')) <= 2 for ws in lines)\n'
                    '\n'
                    '        def try_size(size, no_orphans=False):\n'
                    '            measure, line_height, wrap, pads = make(size)\n'
                    '            lh = size * 1.2\n'
                    '            usable = max(1, rw - pads[0] - pads[2])\n'
                    '            n_lo = max(1, int(math.ceil(measure(words)[3] / usable)))\n'
                    '            for n in range(n_lo, int(rh // lh) + 1):\n'
                    '                out = wrap(n, lh)\n'
                    '                if out is None:\n'
                    '                    continue\n'
                    '                real_lh = line_height(out[0])\n'
                    "                if real_lh > lh:  # the renderer spaces lines by the first line's ink height\n"
                    '                    lh = real_lh\n'
                    '                    out = wrap(n, lh)\n'
                    '                    if out is None or line_height(out[0]) > lh:\n'
                    '                        lh = size * 1.2\n'
                    '                        continue\n'
                    '                if no_orphans and orphaned(out):\n'
                    '                    continue\n'
                    '                return out, n, lh\n'
                    '            return None\n'
                    '\n'
                    '        def largest(hi, no_orphans=False):\n'
                    '            lo, found = min_font_size, None\n'
                    '            while lo <= hi:\n'
                    '                mid = (lo + hi) // 2\n'
                    '                out = try_size(mid, no_orphans)\n'
                    '                if out is not None:\n'
                    '                    found = (mid,) + out\n'
                    '                    lo = mid + 1\n'
                    '                else:\n'
                    '                    hi = mid - 1\n'
                    '            return found\n'
                    '\n'
                    '        best = largest(init_font_size)\n'
                    '        if best is not None and orphaned(best[1]):\n'
                    '            # Like the rectangle fit, keep short words with a neighbour when that costs little size\n'
                    '            tidy = largest(best[0], no_orphans=True)\n'
                    '            if tidy is not None and tidy[0] >= best[0] - max(2, best[0] // 10):\n'
                    '                best = tidy\n'
                    '        if best is not None:\n'
                    '            size, lines, n, lh = best\n'
                    '            if n > 1:\n'
                    '                # Balance the lines: the narrowest width cap that still wraps into n lines\n'
                    '                measure, line_height, wrap, _pads = make(size)\n'
                    '                lo_cap = max(measure([wd])[3] for wd in words)\n'
                    '                hi_cap = max(measure(ws)[3] for ws in lines)\n'
                    '                keep_tidy = not orphaned(lines)\n'
                    '                while lo_cap < hi_cap:\n'
                    '                    cap = (lo_cap + hi_cap) // 2\n'
                    '                    out = wrap(n, lh, cap)\n'
                    '                    if (out is not None and line_height(out[0]) <= lh\n'
                    '                            and not (keep_tidy and orphaned(out))):\n'
                    '                        lines, hi_cap = out, cap\n'
                    '                    else:\n'
                    '                        lo_cap = cap + 1\n'
                    "            return '\\n'.join(' '.join(ws) for ws in lines), size, True\n"
                    '        # Nothing fits even at the minimum size: wrap to the widest line at the minimum size\n'
                    '        measure, _line_height, _wrap, pads = make(min_font_size)\n'
                    '        usable = max(1, rw - pads[0] - pads[2])\n'
                    '        lines, cur = [], []\n'
                    '        for wd in words:\n'
                    '            if cur and measure(cur + [wd])[3] > usable:\n'
                    '                lines.append(cur)\n'
                    '                cur = [wd]\n'
                    '            else:\n'
                    '                cur.append(wd)\n'
                    '        lines.append(cur)\n'
                    "        return '\\n'.join(' '.join(ws) for ws in lines), min_font_size, False\n"
                    '\n'
                    '    def _fit_text_to_region(self, text: str, max_width: int, max_height: int, draw: ImageDraw, region: TextRegion = None, use_as_is: bool = False) -> Tuple[int, List[str]]:\n'
                    '        """Find optimal font size using comic-translate\'s pil_word_wrap algorithm with algorithm-aware adjustments\n'
                    ,
                ),
                (
                    '            init_font_size = base_init\n'
                    '        \n'
                    "        # Use comic-translate's pil_word_wrap algorithm\n"
                    '        wrapped_text, final_font_size = self._pil_word_wrap(\n'
                    ,
                    '            init_font_size = base_init\n'
                    '        \n'
                    '        # Bubble interior found by render_translated_text (_container_render_area): fit lines to it\n'
                    "        container = getattr(region, '_fit_container', None) if (use_as_is and region is not None) else None\n"
                    '\n'
                    "        # Use comic-translate's pil_word_wrap algorithm\n"
                    '        wrapped_text, final_font_size = self._pil_word_wrap(\n'
                    ,
                ),
                (
                    '            init_font_size=init_font_size,\n'
                    '            min_font_size=min_font_size,\n'
                    '            draw=draw\n'
                    '        )\n'
                    '        \n'
                    ,
                    '            init_font_size=init_font_size,\n'
                    '            min_font_size=min_font_size,\n'
                    '            draw=draw,\n'
                    '            container=container\n'
                    '        )\n'
                    '        \n'
                    ,
                ),
                (
                    '            if self.constrain_to_bubble:\n'
                    '                # Re-wrap at target size to check fit\n'
                    '                test_wrapped, _ = self._pil_word_wrap(\n'
                    '                    text=text,\n'
                    '                    font_path=self.selected_font_style or self.font_path,\n'
                    '                    roi_width=usable_width,\n'
                    '                    roi_height=usable_height,\n'
                    '                    init_font_size=target_size,\n'
                    '                    min_font_size=target_size,  # Force this size\n'
                    '                    draw=draw\n'
                    '                )\n'
                    "                test_lines = test_wrapped.split('\\n') if test_wrapped else [text]\n"
                    '                test_height = len(test_lines) * target_size * 1.2\n'
                    '                \n'
                    '                if test_height <= usable_height:\n'
                    '                    final_font_size = target_size\n'
                    '                    lines = test_lines\n'
                    ,
                    '            if self.constrain_to_bubble:\n'
                    '                # Re-wrap at target size to check fit\n'
                    '                if container is not None:\n'
                    '                    test_wrapped, _, test_fits = self._fit_lines_to_container(\n'
                    '                        text, self.selected_font_style or self.font_path, container,\n'
                    '                        target_size, target_size, draw)\n'
                    '                else:\n'
                    '                    test_wrapped, _ = self._pil_word_wrap(\n'
                    '                        text=text,\n'
                    '                        font_path=self.selected_font_style or self.font_path,\n'
                    '                        roi_width=usable_width,\n'
                    '                        roi_height=usable_height,\n'
                    '                        init_font_size=target_size,\n'
                    '                        min_font_size=target_size,  # Force this size\n'
                    '                        draw=draw\n'
                    '                    )\n'
                    '                    test_fits = None\n'
                    "                test_lines = test_wrapped.split('\\n') if test_wrapped else [text]\n"
                    '                test_height = len(test_lines) * target_size * 1.2\n'
                    '                \n'
                    '                if test_fits if test_fits is not None else test_height <= usable_height:\n'
                    '                    final_font_size = target_size\n'
                    '                    lines = test_lines\n'
                    ,
                ),
            ],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga editor Format toolbar: a Word-style row at the top of the image preview (font, size, bold / "
                "italic / underline, alignment, text colour, reset) styles the selected boxes; the styles are "
                "stored per box in the page state (region_styles) and drawn by the renderer",
        "source": {
            "src/manga_integration.py": [_FORMAT_TOOLBAR_INSTALL],
            "src/ImageRenderer.py": [_FORMAT_CONTEXT_MENU],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga glossary workflow: the OCR-only glossary pass detects a tall webtoon strip chunk by chunk "
                "(the webtoon path's chunking and overlap de-duplication, now _detect_webtoon_chunk_regions), "
                "so the regions translation reuses are the chunked ones; the chunk shift also moves "
                "bubble_bounds and a merged region's original_regions, and the overlap de-duplication "
                "finds the previous chunk's copy (it skipped every earlier region before)",
        "source": {
            "src/manga_translator.py": [
                (
                    '            # owns its full translate/render path, so skip that shortcut for\n'
                    '            # OCR-only and precomputed-region glossary passes.\n'
                    '            used_precomputed_regions = precomputed_regions is not None\n'
                    ,
                    '            # owns its full translate/render path, so skip that shortcut for\n'
                    '            # OCR-only and precomputed-region glossary passes. An OCR-only pass\n'
                    '            # still detects a webtoon strip chunk by chunk (same regions as the\n'
                    '            # webtoon path), and the translation pass reuses those regions.\n'
                    '            webtoon_chunk_ocr = False\n'
                    '            used_precomputed_regions = precomputed_regions is not None\n'
                ),
                (
                    "                webtoon_mode = self.manga_settings.get('advanced', {}).get('webtoon_mode', 'auto')\n"
                    '                if (not ocr_only and not used_precomputed_regions and\n'
                    ,
                    "                webtoon_mode = self.manga_settings.get('advanced', {}).get('webtoon_mode', 'auto')\n"
                    '                if (ocr_only and not used_precomputed_regions and\n'
                    "                        format_info['is_webtoon'] and webtoon_mode in ('auto', 'force')):\n"
                    '                    self._log("🔄 Webtoon mode active - will OCR in chunks for better OCR")\n'
                    '                    webtoon_chunk_ocr = True\n'
                    '                if (not ocr_only and not used_precomputed_regions and\n'
                ),
                (
                    '                    preserve_translations=used_imported_regions,\n'
                    '                )\n'
                    '            else:\n'
                    '                self._log(f"📍 [STEP 1] Text Detection Phase")\n'
                    ,
                    '                    preserve_translations=used_imported_regions,\n'
                    '                )\n'
                    '            elif webtoon_chunk_ocr:\n'
                    '                self._log(f"📍 [STEP 1] Text Detection Phase (webtoon chunks)")\n'
                    '                _, regions, num_chunks = self._detect_webtoon_chunk_regions(image_path)\n'
                    "                result['format_info']['chunks_processed'] = num_chunks\n"
                    '                if self._check_stop():\n'
                    '                    # A stopped chunk OCR returns no regions for the chunks it skipped:\n'
                    '                    # do not save or hand over that partial page as its OCR.\n'
                    "                    result['interrupted'] = True\n"
                    '                    self._log("⏹️ Webtoon chunk OCR stopped", "warning")\n'
                    '                    return result\n'
                    '            else:\n'
                    '                self._log(f"📍 [STEP 1] Text Detection Phase")\n'
                ),
                (
                    '        import cv2\n'
                    '        import numpy as np\n'
                    '        from PIL import Image as PILImage\n'
                    '        \n'
                    ,
                    '        import cv2\n'
                    '        \n'
                ),
                (
                    '            \n'
                    '            # Load the image\n'
                    '            image = cv2.imread(image_path)\n'
                    '            if image is None:\n'
                    '                pil_image = open_image(image_path)\n'
                    '                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)\n'
                    '            \n'
                    '            height, width = image.shape[:2]\n'
                    '            \n'
                    '            # Get chunk settings from config\n'
                    "            chunk_height = self.manga_settings.get('preprocessing', {}).get('chunk_height', 1000)\n"
                    "            chunk_overlap = self.manga_settings.get('preprocessing', {}).get('chunk_overlap', 100)\n"
                    '            \n'
                    '            self._log(f"   Image dimensions: {width}x{height}", "info")\n'
                    '            self._log(f"   Chunk height: {chunk_height}px, Overlap: {chunk_overlap}px", "info")\n'
                    '            \n'
                    '            # Calculate number of chunks needed\n'
                    '            effective_chunk_height = chunk_height - chunk_overlap\n'
                    '            num_chunks = max(1, (height - chunk_overlap) // effective_chunk_height + 1)\n'
                    '            \n'
                    '            self._log(f"   Will process in {num_chunks} chunks", "info")\n'
                    '            \n'
                    '            # Process each chunk\n'
                    '            all_regions = []\n'
                    '            chunk_offsets = []\n'
                    '            \n'
                    '            for i in range(num_chunks):\n'
                    '                # Calculate chunk boundaries\n'
                    '                start_y = i * effective_chunk_height\n'
                    '                end_y = min(start_y + chunk_height, height)\n'
                    '                \n'
                    "                # Make sure we don't miss the bottom part\n"
                    '                if i == num_chunks - 1:\n'
                    '                    end_y = height\n'
                    '                \n'
                    '                self._log(f"\\n   📄 Processing chunk {i+1}/{num_chunks} (y: {start_y}-{end_y})", "info")\n'
                    '                \n'
                    '                # Extract chunk\n'
                    '                chunk = image[start_y:end_y, 0:width]\n'
                    '                \n'
                    '                # Save chunk temporarily for OCR\n'
                    '                import tempfile\n'
                    "                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:\n"
                    '                    chunk_path = tmp.name\n'
                    '                    cv2.imwrite(chunk_path, chunk)\n'
                    '                \n'
                    '                try:\n'
                    '                    # Detect text in this chunk\n'
                    '                    chunk_regions = self.detect_text_regions(chunk_path)\n'
                    '                    \n'
                    '                    # Adjust region coordinates to full image space\n'
                    '                    for region in chunk_regions:\n'
                    '                        # Adjust bounding box\n'
                    '                        x, y, w, h = region.bounding_box\n'
                    '                        region.bounding_box = (x, y + start_y, w, h)\n'
                    '                        \n'
                    '                        # Adjust vertices if present\n'
                    "                        if hasattr(region, 'vertices') and region.vertices:\n"
                    '                            adjusted_vertices = []\n'
                    '                            for vx, vy in region.vertices:\n'
                    '                                adjusted_vertices.append((vx, vy + start_y))\n'
                    '                            region.vertices = adjusted_vertices\n'
                    '                        \n'
                    '                        # Mark which chunk this came from (for deduplication)\n'
                    '                        region.chunk_index = i\n'
                    '                        region.chunk_y_range = (start_y, end_y)\n'
                    '                    \n'
                    '                    all_regions.extend(chunk_regions)\n'
                    '                    chunk_offsets.append(start_y)\n'
                    '                    \n'
                    '                    self._log(f"   Found {len(chunk_regions)} text regions in chunk {i+1}", "info")\n'
                    '                    \n'
                    '                finally:\n'
                    '                    # Clean up temp file\n'
                    '                    import os\n'
                    '                    if os.path.exists(chunk_path):\n'
                    '                        os.remove(chunk_path)\n'
                    '            \n'
                    '            # Remove duplicate regions from overlapping areas\n'
                    '            self._log(f"\\n   🔍 Deduplicating regions from overlaps...", "info")\n'
                    '            unique_regions = self._deduplicate_chunk_regions(all_regions, chunk_overlap)\n'
                    '            \n'
                    '            self._log(f"   Total regions: {len(all_regions)} → {len(unique_regions)} after deduplication", "info")\n'
                    '            \n'
                    ,
                    '            \n'
                    '            image, unique_regions, num_chunks = self._detect_webtoon_chunk_regions(image_path)\n'
                    '            \n'
                ),
                (
                    "            result['errors'].append(error_msg)\n"
                    '            return result\n'
                    '\n'
                    '    def _deduplicate_chunk_regions(self, regions: List, overlap_height: int) -> List:\n'
                    ,
                    "            result['errors'].append(error_msg)\n"
                    '            return result\n'
                    '\n'
                    '    def _detect_webtoon_chunk_regions(self, image_path: str):\n'
                    '        """Detect the text of a tall webtoon strip chunk by chunk.\n'
                    '\n'
                    '        Returns ``(image, regions, num_chunks)``: the loaded BGR page, the regions in\n'
                    '        full-image coordinates with the overlap duplicates removed, and the chunk count.\n'
                    '        The webtoon translation path and the glossary OCR-only pass both call this, so the\n'
                    '        regions a glossary pass hands to translation are the ones translation would detect.\n'
                    '        """\n'
                    '        import cv2\n'
                    '        import numpy as np\n'
                    '        \n'
                    '        # Load the image\n'
                    '        image = cv2.imread(image_path)\n'
                    '        if image is None:\n'
                    '            pil_image = open_image(image_path)\n'
                    '            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)\n'
                    '        \n'
                    '        height, width = image.shape[:2]\n'
                    '        \n'
                    '        # Get chunk settings from config\n'
                    "        chunk_height = self.manga_settings.get('preprocessing', {}).get('chunk_height', 1000)\n"
                    "        chunk_overlap = self.manga_settings.get('preprocessing', {}).get('chunk_overlap', 100)\n"
                    '        \n'
                    '        self._log(f"   Image dimensions: {width}x{height}", "info")\n'
                    '        self._log(f"   Chunk height: {chunk_height}px, Overlap: {chunk_overlap}px", "info")\n'
                    '        \n'
                    '        # Calculate number of chunks needed\n'
                    '        effective_chunk_height = chunk_height - chunk_overlap\n'
                    '        num_chunks = max(1, (height - chunk_overlap) // effective_chunk_height + 1)\n'
                    '        \n'
                    '        self._log(f"   Will process in {num_chunks} chunks", "info")\n'
                    '        \n'
                    '        # Process each chunk\n'
                    '        all_regions = []\n'
                    '        chunk_offsets = []\n'
                    '        \n'
                    '        for i in range(num_chunks):\n'
                    '            # Calculate chunk boundaries\n'
                    '            start_y = i * effective_chunk_height\n'
                    '            end_y = min(start_y + chunk_height, height)\n'
                    '            \n'
                    "            # Make sure we don't miss the bottom part\n"
                    '            if i == num_chunks - 1:\n'
                    '                end_y = height\n'
                    '            \n'
                    '            self._log(f"\\n   📄 Processing chunk {i+1}/{num_chunks} (y: {start_y}-{end_y})", "info")\n'
                    '            \n'
                    '            # Extract chunk\n'
                    '            chunk = image[start_y:end_y, 0:width]\n'
                    '            \n'
                    '            # Save chunk temporarily for OCR\n'
                    '            import tempfile\n'
                    "            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:\n"
                    '                chunk_path = tmp.name\n'
                    '                cv2.imwrite(chunk_path, chunk)\n'
                    '            \n'
                    '            try:\n'
                    '                # Detect text in this chunk\n'
                    '                chunk_regions = self.detect_text_regions(chunk_path)\n'
                    '                \n'
                    '                # Adjust region coordinates to full image space: the box and polygon of the\n'
                    "                # region and of a merged region's lines (create_text_mask masks those), and the\n"
                    '                # RT-DETR bubble it is masked and rendered in\n'
                    '                shifted_ids = set()\n'
                    '                for region in chunk_regions:\n'
                    "                    for shifted in [region] + list(getattr(region, 'original_regions', None) or []):\n"
                    '                        if id(shifted) in shifted_ids:\n'
                    '                            continue\n'
                    '                        shifted_ids.add(id(shifted))\n'
                    '                        # Adjust bounding box\n'
                    '                        x, y, w, h = shifted.bounding_box\n'
                    '                        shifted.bounding_box = (x, y + start_y, w, h)\n'
                    '                        \n'
                    '                        # Adjust vertices if present\n'
                    "                        if hasattr(shifted, 'vertices') and shifted.vertices:\n"
                    '                            shifted.vertices = [(vx, vy + start_y) for vx, vy in shifted.vertices]\n'
                    '                    \n'
                    "                    bubble_bounds = getattr(region, 'bubble_bounds', None)\n"
                    '                    if bubble_bounds:\n'
                    '                        bx, by, bw, bh = bubble_bounds\n'
                    '                        region.bubble_bounds = (bx, by + start_y, bw, bh)\n'
                    '                    \n'
                    '                    # Mark which chunk this came from (for deduplication)\n'
                    '                    region.chunk_index = i\n'
                    '                    region.chunk_y_range = (start_y, end_y)\n'
                    '                \n'
                    '                all_regions.extend(chunk_regions)\n'
                    '                chunk_offsets.append(start_y)\n'
                    '                \n'
                    '                self._log(f"   Found {len(chunk_regions)} text regions in chunk {i+1}", "info")\n'
                    '                \n'
                    '            finally:\n'
                    '                # Clean up temp file\n'
                    '                import os\n'
                    '                if os.path.exists(chunk_path):\n'
                    '                    os.remove(chunk_path)\n'
                    '        \n'
                    '        # Remove duplicate regions from overlapping areas\n'
                    '        self._log(f"\\n   🔍 Deduplicating regions from overlaps...", "info")\n'
                    '        unique_regions = self._deduplicate_chunk_regions(all_regions, chunk_overlap)\n'
                    '        \n'
                    '        self._log(f"   Total regions: {len(all_regions)} → {len(unique_regions)} after deduplication", "info")\n'
                    '        \n'
                    '        return image, unique_regions, num_chunks\n'
                    '\n'
                    '    def _deduplicate_chunk_regions(self, regions: List, overlap_height: int) -> List:\n'
                ),
                (
                    '                for j, region2 in enumerate(regions):\n'
                    '                    if j >= i or j in used_indices:\n'
                    '                        continue\n'
                    '                    \n'
                    ,
                    '                for j, region2 in enumerate(regions):\n'
                    '                    # Every earlier region is in used_indices, kept ones too, and OCR jitter can\n'
                    "                    # sort the previous chunk's copy after this one: compare with all the others.\n"
                    '                    if j == i:\n'
                    '                        continue\n'
                    '                    \n'
                ),
            ],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga Translate after switching the Local Model (custom-image-edit -> lama_onnx): the Local Model "
                "switch refreshes Skip Inpainter's in-use lock at once; Translate waits for its concurrent "
                "inpainting (first-run download / load) before rendering, and a failed image edit request "
                "returns its pool checkout (manga_editor_core, test_manga_editor_core EDITS)",
        "source": {
            "src/manga_integration.py": [_SKIP_LOCK_SWITCH_IMAGE_EDIT, _SKIP_LOCK_SWITCH_LOCAL],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga OCR prompt: the empty-answer rule (rule 8 and the last example line) covers illegible text too "
                "(\"no visible / illegible\"); saved prompts with the previous default lines are migrated",
        "source": {
            "src/manga_integration.py": [_OCR_PROMPT_ILLEGIBLE_DEFAULT, _OCR_PROMPT_ILLEGIBLE_MIGRATE],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga Glossary Workflow: 'Include page images in glossary generation' (manga_glossary_include_images, "
                "default on) under Generate Glossary; Generate Glossary and a batch run's glossary step send the page "
                "images with the OCR text in page batches (tall pages chunked like translation, entries merged and "
                "de-duplicated); a model without image input falls back to the text-only request",
        "source": {
            "src/manga_integration.py": _MANGA_GLOSSARY_PAGE_IMAGES,
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Image runs: the image progress tracker follows each run's output folder (set_output_dir when it "
                "changes), so every image / folder output keeps its own translation_progress.json",
        "source": {
            # the image runner lived in translator_gui before U7
            "src/translator_gui.py": [_IMAGE_PROGRESS_FOLLOWS_RUN],
            "src/image_job.py": [_IMAGE_PROGRESS_FOLLOWS_RUN],
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Image gen/edit key pool used everywhere: export_key_pool_env also exports USE_INPAINTER_KEYS / "
                "INPAINTER_API_KEYS and USE_TTS_KEYS / TTS_API_KEYS; Generate from prompt (no input file) sets "
                "up the Image gen/edit and Audio/TTS pools like an input image and sends an image request with "
                "the 'image_generation' context, so the pool key is used instead of the main key "
                "(image_job._configure_media_output_key_pools; test_image_job EDITS)",
        "source": {
            "src/translator_gui.py": [_MEDIA_POOL_ENV_TG, _generative_pools(_MEDIA_POOL_SETUP), _GENERATIVE_SEND],
            "src/key_pools.py": [_MEDIA_POOL_ENV_KEY_POOLS],
            "src/image_job.py": [_generative_pools(_MEDIA_POOL_CALL), _GENERATIVE_SEND, _IMAGE_FILE_POOLS_CALL,
                                 _MEDIA_POOL_HELPER],
        },
        # new env names only: recorded values of the existing ones do not change
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga text requests (the editor's Translate / Translate This Text clients) ignore the main tab's "
                "Output mode: an 'image' Output mode no longer turns a manga translation into an image request",
        "source": {
            "src/ImageRenderer.py": _MANGA_TEXT_ONLY_EDITOR,
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Import OCR: 'Merge with detection' (off by default) keeps the page's detected / existing regions and "
                "adds the imported ones, an imported region replacing every region it overlaps by half the smaller box",
        "source": {
            "src/manga_integration.py": _OCR_IMPORT_MERGE,
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Manga glossary matching reads the page text as OCR text (its Korean word spacing is unreliable): "
                "a 2+ syllable glossary term glued to its neighbours (부록의마왕) is in the page",
        "source": {
            "src/ImageRenderer.py": _MANGA_GLOSSARY_OCR_MATCH_EDITOR,
        },
        "values": {},
    },
    {
        "date": "2026-11-06",
        "what": "Built-in Manga_KR profile keeps Korean honorifics and family address terms in romanized form like the "
                "Korean text profiles; a saved Manga_KR that still holds the old default lines is migrated",
        "source": {
            "src/owner_state.py": _MANGA_KR_HONORIFICS,
            "src/translator_gui.py": _MANGA_KR_HONORIFICS,
        },
        "values": {"Manga_KR": (_MANGA_KR_OLD, _MANGA_KR_NEW)},
    },
    {
        "date": "2026-11-06",
        "what": "Manga glossary prompt: page-image rules (fix OCR from the image, gender from the art) and name rules "
                "(titles used as names, the story's own names, romanized honorifics); a saved prompt that still ends "
                "with the old default's last line gets them",
        "source": {
            "src/manga_integration.py": _MANGA_GLOSSARY_IMAGE_NAME_RULES,
        },
        "values": {},
    },
    {
        "date": "2026-11-07",
        "what": "Compiled Windows builds have no console: the manga tab shows console_window's runtime console "
                "while a run is active (start) and hides it when the run resets (Other Settings Show CMD keeps it)",
        "source": {
            "src/manga_integration.py": _CONSOLE_WINDOW_MANGA,
        },
        "values": {},
    },
    {
        "date": "2026-11-07",
        "what": "QA Scanner settings: the 'Check glossary names' label texts moved into glossary_name_check."
                "NER_TOGGLE_LABELS (same text), shared with Glossarion Mobile's QA Scanner Settings tile",
        "source": {
            "src/QA_Scanner_GUI.py": [(
                '            ner_labels = {\n'
                '                "installed": "Check glossary names with NER (Experimental)",\n'
                '                "missing": "Check glossary names (Experimental) — NER not installed (downloads on first use)",\n'
                '                "unsupported": "Check glossary names (Experimental) — NER not supported in this build",\n'
                '                "off": "Check glossary names (Experimental) — NER off",\n'
                '            }\n',
                "            ner_labels = name_check.NER_TOGGLE_LABELS  # mobile's QA Scanner Settings tile shows the same text\n",
            )],
        },
        "values": {},
    },
    {
        "date": "2026-11-07",
        "what": "Manga Custom Image Edit Prompt dialog: a prompt profile row like the review / glossary prompt "
                "profiles (review_dialog's shared row over manga_env.ImageEditPromptProfiles): Default (the prompt "
                "used before profiles; a saved custom prompt stays in it), the built-in Translate SFX and Translate "
                "SFX + background text, and the user's own. Boxes the user draws in the editor are marked "
                "user_drawn (saved with the box) and Translate cleans / image-edits them without OCR text",
        "source": {
            "src/manga_integration.py": _IMAGE_EDIT_PROMPT_PROFILES,
            "src/manga_image_preview.py": _USER_DRAWN_BOXES_PREVIEW,
        },
        "values": {},
    },
    {
        "date": "2026-11-07",
        "what": "Manga Inpaint Method: separate radios Replicate (Bad) / Local (Light-weight) / API (Recommended) / "
                "Hybrid (Trash). API is the custom image edit inpainter (manga_inpaint_method 'api', no model "
                "dropdown); the Local dropdown no longer lists custom-image-edit. An old 'local' + "
                "'custom-image-edit' config moves to 'api' (manga_models.migrate_inpaint_method), and every reader "
                "of the method treats 'api' as the custom image edit inpainter",
        "source": {
            "src/manga_integration.py": _INPAINT_API_MANGA_INTEGRATION + _INPAINT_API_RUNNER,
            "src/ImageRenderer.py": _INPAINT_API_EDITOR,
        },
        "values": {},
    },
    {
        "date": "2026-11-08",
        "what": "Manga runs no longer show the console (owner: 'just remove it'); only Other Settings Show CMD "
                "opens it. The 2026-11-07 start / reset toggles and _set_console_visible are removed again",
        "source": {
            "src/manga_integration.py": [(new, old) for old, new in _CONSOLE_WINDOW_MANGA],
        },
        "values": {},
    },
    {
        "date": "2026-11-08",
        "what": "Manga preview: Del deletes the selected box while Manual Editing is on (the Delete "
                "Selected button's action, not while typing in a box); Import / Export fill the toolbar "
                "row's height",
        "source": {
            "src/manga_image_preview.py": _DEL_KEY_IMPORT_EXPORT_PREVIEW,
        },
        "values": {},
    },
    {
        "date": "2026-11-13",
        "what": "search/gemini overhaul: the route opens google.com in AI Mode and attaches the text and images "
                "as files, so the Other Settings section 'Gemini Free Browser Chunking' (adaptive sub-chunks, HTML "
                "text-node transport and the ten chunk tuning fields) is removed with its 12 config keys, owner "
                "vars and GEMINI_FREE_* env vars. A config that still holds the keys keeps them untouched",
        "source": {relpath: _removed_rows(rows) for relpath, rows in _GEMINI_CHUNKING_REMOVED_ROWS.items()},
        "values": {},
        "removes": tuple(_GEMINI_CHUNKING_KEYS) + tuple(key + "_var" for key in _GEMINI_CHUNKING_KEYS)
                   + tuple(key.upper() for key in _GEMINI_CHUNKING_KEYS),
    },
    {
        "date": "2026-11-13",
        "what": "New glossary setting 'Enable Relations Tracker' (glossary_enable_relations_tracker, off) with 'Earlier relations shown' (glossary_relations_history_mode, previous) and the relations field name "
                "(glossary_relations_field_name, relations): the glossary AI returns a relations column that is "
                "kept in a *_relations_tracker.json sidecar, never in the glossary file",
        "source": _RELATIONS_TRACKER_ROWS,
        "values": {},
    },
    {
        "date": "2026-11-13",
        "what": "New glossary setting glossary_placeholder_texts ({}: none stored), sent as GLOSSARY_PLACEHOLDER_TEXTS "
                "(JSON): the texts the user wrote for the editable glossary prompt placeholders "
                "({relations_instruction}, {subject_tracking_instruction}, {description_example}, ...) in Glossary "
                "Manager > right-click 'Available placeholders' > 'Edit placeholder text…'. With nothing stored "
                "every prompt is unchanged",
        "source": _PLACEHOLDER_TEXTS_ROWS,
        "values": {},
    },
    {
        "date": "2026-11-13",
        "what": "New Minimal glossary setting 'Include Relations Column' (glossary_minimal_include_relations, off), "
                "sent as GLOSSARY_MINIMAL_INCLUDE_RELATIONS: Minimal Glossary Generation asks for and records "
                "relations only with it and Enable Relations Tracker on (it used to follow the tracker alone)",
        "source": _MINIMAL_RELATIONS_ROWS,
        "values": {},
    },
    {
        "date": "2026-11-13",
        "what": "A fresh config (no config.json, or one Reset Settings to Defaults wrote) starts with Enable "
                "Relations Tracker and Include Relations Column on and the relations column in Custom Fields "
                "(translator_gui marks it, relations_tracking.apply_fresh_defaults); existing configs are not "
                "touched, so only the fresh_install scenario changes",
        "source": _FRESH_RELATIONS_ROWS,
        "values": {},
    },
)


def shared_subtitle_lists(text: str) -> str:
    """Legacy source with its copies of the subtitle extensions as ``subtitle_processor.SUBTITLE_EXTENSIONS``
    (2026-11-02 .ssa/.vtt/.sbv). For the moved-verbatim guards, whose legacy text is compared, never run."""
    import re

    for pattern, replacement in (
        (r"""(['"])\.srt\1, \1\.ass\1, \1\.lrc\1, \1\.vtt\1""", "*SUBTITLE_EXTENSIONS"),
        (r"""[({](['"])\.srt\1, \1\.ass\1, \1\.lrc\1[)}]""", "SUBTITLE_EXTENSIONS"),
        (r"""(['"])\.srt\1, \1\.ass\1, \1\.lrc\1""", "*SUBTITLE_EXTENSIONS"),
    ):
        text = re.sub(pattern, replacement, text)
    text = re.sub(
        r""" *"\.srt": \{"\.srt": 160, "\.txt": 70\},\n *"\.ass": \{"\.ass": 160, "\.txt": 70\},\n( *)"\.lrc": \{"\.lrc": 160, "\.txt": 70\},""",
        lambda m: m.group(1) + '**{extension: {extension: 160, ".txt": 70} for extension in SUBTITLE_EXTENSIONS},', text)
    for old in ('"*.xhtml *.xml *.json *.csv *.tsv *.srt *.ass *.lrc *.vtt *.log *.png *.jpg "',
                '"*.csv *.tsv *.srt *.ass *.lrc *.vtt *.log);;EPUB files (*.epub);;PDF files (*.pdf);;"'):
        text = text.replace(old, "f" + old.replace("*.srt *.ass *.lrc *.vtt", "{SUBTITLE_FILE_PATTERNS}"))
    return text.replace("legacy support for markup/subtitle/log files", "legacy support for markup/log files")


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
    """A loaded golden (nested dicts / lists) with the approved recorded values changed, the
    approved new keys inserted and the keys of approved removed settings dropped."""
    values = {}
    inserts = []
    removes = set()
    for change in APPROVED:
        values.update(change["values"])
        inserts.extend(change.get("inserts", ()))
        removes.update(change.get("removes", ()))

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
            # "removes": a removed setting's recorded key (config key, owner var, env var) is dropped
            return {key: (_changed(walk(value, key), *values[key]) if key in values else walk(value, key))
                    for key, value in node.items() if key not in removes}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(walk(item) for item in node)
        return node

    return apply_epub_compiler_log_change(walk(data))


def apply_epub_compiler_log_change(result):
    # The EPUB converter reports success only when it creates or updates an EPUB.
    # The golden's fake compiler returns without producing a file.
    calls = result.get("entries", {}).get("epub_compile", {}).get("calls", [])
    success = ["append_log", ["✅ EPUB Converter completed successfully!"], {}]
    missing = ["append_log", ["⚠️ EPUB file was not created. Check the logs for details."], {}]
    for index, (call, following) in enumerate(zip(calls, calls[1:])):
        if call == success and following == missing:
            del calls[index]
            break
    return result
