"""translation_pipeline: the desktop translation and glossary pipelines, shared by TranslatorGUI and mobile.

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). The methods below moved
verbatim out of ``TranslatorGUI`` (``translator_gui.py`` @ 1719fb59), which inherits
``TranslationPipelineMixin`` as its first base; ``HeadlessOwner`` runs the same code on
mobile:

* ``run_translation_thread`` (29015-29816) is split at its worker thread:
  ``_prepare_translation_run(files=None) -> RunRequest`` is the run set-up after the
  Run-button preflight (29081-29336: input check ... existing QA-failure summary) and
  ``_translation_worker(request)`` is the body of its ``simple_thread_target`` closure
  (29339-29798: module loading, archive inputs, the Balanced/Full pre-translation glossary
  pass with retries, the require-complete gate and the Direct Text approval,
  ``run_translation_direct`` and the multipass refinement follow-up, the post-translation
  QA trigger, the end-of-run reset). The desktop ``run_translation_thread`` keeps the
  preflight (glossary-run guard, Parallel EPUB Pair notice, Run-as-Stop toggle, stop
  clean-up wait, double click after a graceful stop, model check) and starts a thread
  running ``_translation_worker(self._prepare_translation_run())``;
* ``run_translation_direct`` (29945-30433): the per-file loop;
* ``_clear_automatic_glossary_for_non_epub_selection`` (24454) and the QA-failure collection
  and multipass refinement planning (``_flatten_translation_qa_issue_text`` ...
  ``_log_translation_qa_failure_summary``, 24687-25103);
* ``run_glossary_extraction_direct`` (32643-33030) and the image-folder glossary
  (``_process_image_folder_for_glossary`` ... ``_call_api_with_interrupt``, 33032-33821);
* glossary auto-loading / auto-mapping (``auto_load_glossary_for_file``,
  ``_auto_load_glossary_after_extraction``, ``_autofill_glossary_for_current_selection``,
  ``_sync_automapped_glossaries_to_output`` and helpers) and the input-selection helpers
  both pipelines use (OPF order, special files, Windows-safe input names);
* ``_glossary_editor_input_sources`` moved from ``GlossaryManager_GUI.GlossaryManagerMixin``
  (which re-exports it).

Edits made while moving (everything else is byte-for-byte):

* the closure became a method (body dedented one level); the set-up ends in
  ``return RunRequest(...)`` (the closure captured no locals: it reads the owner);
* the lazily loaded module globals ``translation_stop_flag`` / ``glossary_main`` /
  ``glossary_stop_flag`` are read through the ``_backend_entry`` hook at the same point;
* Qt signals are the ``_ui_request`` hook (``trigger_qa_scan``, ``thread_complete``,
  ``input_files_updated``, and the blocking ``direct_text_glossary_approval`` question);
  the "no file selected" message box is the ``_ui_message`` hook;
* ``TranslatorGUI.<name>(self, ...)`` class-qualified calls name the mixin that now
  defines ``<name>`` (``TranslationPipelineMixin`` / ``RunEnvMixin``);
* ``_InputOutputDialog._IMAGE_ATTACHMENT_EXTENSIONS`` is ``IMAGE_ATTACHMENT_EXTENSIONS``
  (the dialog's class attribute is now this set);
* (U3 fix pass) ``_translation_worker`` returns its outcome: ``return False`` on the
  closure's early returns and after its caught exception, ``return translation_completed``
  at the end of its ``try`` (the desktop thread ignores the value); the Library raw-input
  registry write of the set-up is the ``_record_library_raw_inputs`` hook (the desktop
  override keeps the epub_library import; the GUI-free default skips the registry, which
  epub_library builds only).

``PipelineHooksMixin`` holds the GUI-free defaults of the hooks and of the desktop GUI
methods the moved code calls (TranslatorGUI's own methods take precedence over every
mixin, so the desktop keeps its originals). The image / RPG Maker / generative-prompt
runners ``run_translation_direct`` dispatches to moved out of TranslatorGUI in U7:
``TranslationPipelineMixin`` inherits ``image_job.ImageJobMixin`` and
``rpgmaker_job.RpgMakerJobMixin`` (they replaced the U3 placeholders of
``PipelineHooksMixin``). U7 edit of ``run_translation_direct``: a folder registered by the
mobile RPG Maker entry (``rpgmaker_job.RPGMAKER_GAME_INPUTS_ATTR``) is dispatched like a
game ``.exe``; nothing on the desktop registers one.

Mobile composition (JobService, one job at a time)::

    with job_runner.job_process_state(host.log):
        stop_control.reset_for_new_run('translation')
        request = owner._prepare_translation_run(files)
        if request is not None:
            owner._translation_worker(request)

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field

from app_paths import _get_app_dir
from epub_package import find_epub_opf_member, find_opf_path
from image_job import ImageJobMixin
from job_runner import JobHooksMixin
from rpgmaker_job import RPGMAKER_GAME_INPUTS_ATTR, RpgMakerJobMixin
from run_env import RunEnvMixin
from stop_control import clear_client_cancellation, make_run_id, prepare_glossary_stop_file, reset_stop_env
from title_tag_translation import DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT

__all__ = [
    "GlossaryPipelineMixin",
    "IMAGE_ATTACHMENT_EXTENSIONS",
    "PIPELINE_HOOKS",
    "PipelineHooksMixin",
    "RunRequest",
    "TranslationPipelineMixin",
    "UI_QUESTION_FIELDS",
    "UNSHARED_UI_REQUESTS",
]

#: Image inputs a run groups as media (the Direct Text dialog's attachment image types).
IMAGE_ATTACHMENT_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tif', '.tiff',
    '.svg', '.ico', '.heic', '.heif', '.avif', '.jxl',
}

#: ``_ui_request`` kinds that are blocking questions: positional arguments -> fields. The
#: last field is the desktop handshake ``{"event": threading.Event, "accepted": bool}``
#: the worker waits on; the GUI-free default answers it from ``host.ask(kind, **fields)``.
UI_QUESTION_FIELDS = {
    "direct_text_glossary_approval": ("path", "request"),
}

#: ``_ui_request`` kinds whose desktop handler (a GUI method) is not shared yet: the GUI-free
#: default logs this line before emitting the event, so the log never promises work that
#: no front end runs. ``trigger_qa_scan``: the QA scanner (QAScannerMixin.run_qa_scan) moves in U6.
UNSHARED_UI_REQUESTS = {
    "trigger_qa_scan": "⚠️ The post-translation QA scan is not available in this build yet; skipped",
}

#: Every name PipelineHooksMixin defines (desktop hooks / GUI method defaults).
PIPELINE_HOOKS = (
    "_ui_request",
    "_ui_message",
    "_lazy_load_modules",
    "_attach_gui_logging_handlers",
    "_create_watchdog_snapshot",
    "_start_autoscroll_delay",
    "_update_manual_glossary_status",
    "_record_library_raw_inputs",
)


@dataclass
class RunRequest:
    """One prepared translation run (``_prepare_translation_run``'s result).

    The worker reads the owner, exactly like the desktop closure did; these fields
    report what was prepared to the caller (job adapters, logs, tests).
    """

    files: list = field(default_factory=list)
    run_id: str = ''
    qa_resolution_request: dict = None
    multipass_qa_refinement: bool = False
    refinement_mode: str = ''
    followup_translation: bool = False
    metadata_only: bool = False
    single_chapter_filter: str = None
    direct_text: bool = False
    graceful_stop: bool = False
    wait_for_chunks: bool = True


class PipelineHooksMixin(JobHooksMixin):
    """GUI-free defaults of the hooks and desktop GUI methods the moved pipelines call.

    ``TranslatorGUI`` defines every one of these in its own class body (the original Qt
    code: log panel, watchdog bar, glossary status row, message boxes, module loader with
    globals), which wins over all its bases; ``HeadlessOwner`` uses these defaults and
    reports through ``self.host`` (a ``job_runner.JobHost``).
    """

    def _ui_request(self, kind, *args, **data):
        """Blocking questions (``UI_QUESTION_FIELDS``) go to ``host.ask``; the rest to ``JobHooksMixin``.

        Desktop: ``<kind>_signal.emit(*args)``; the GUI answers a question by setting the
        handshake's ``accepted`` and its Event. Here ``host.ask(kind, **fields)`` blocks
        until the user answers (no host, or an error: declined) and the handshake is
        completed with the answer, so the waiting worker resumes at once.
        """
        names = UI_QUESTION_FIELDS.get(kind)
        if names is None:
            unshared = UNSHARED_UI_REQUESTS.get(kind)
            if unshared:
                try:
                    self.append_log(unshared)
                except Exception:
                    pass
            return super()._ui_request(kind, *args, **data)
        fields = dict(zip(names, args))
        fields.update(data)
        request = fields.pop(names[-1], None)
        host = getattr(self, 'host', None)
        ask = getattr(host, 'ask', None)
        answer = False
        try:
            if callable(ask):
                answer = ask(kind, **fields)
        except Exception as exc:
            try:
                self.append_log(f"⚠️ Could not ask for {kind.replace('_', ' ')}: {exc}")
            except Exception:
                pass
            answer = False
        accepted = bool(answer)
        if isinstance(request, dict):
            request['accepted'] = accepted
            event = request.get('event')
            if event is not None:
                event.set()
        return accepted

    def _ui_message(self, level, title, text):
        """A message box (desktop: ``QMessageBox.<level>(self, title, text)``): log + ``message`` event."""
        try:
            self.append_log(f"{'❌' if level == 'critical' else '⚠️'} {title}: {text}")
        except Exception:
            pass
        host = getattr(self, 'host', None)
        emit = getattr(host, 'emit', None)
        if callable(emit):
            emit('message', level=level, title=title, text=text)
        return None

    def _lazy_load_modules(self, splash_callback=None):
        """Import the backend entry points once (desktop: binds translator_gui's globals, splash, QTimer)."""
        if getattr(self, '_modules_loaded', False):
            return True
        names = ('translation_main', 'glossary_main', 'fallback_compile_epub')
        loaded = [name for name in names if self._backend_entry(name) is not None]
        self._modules_loaded = True
        self._modules_loading = False
        if len(loaded) == len(names):
            self.append_log(f"✅ Loaded {len(loaded)}/{len(names)} modules successfully")
            return True
        self.append_log(f"⚠️ Loaded {len(loaded)}/{len(names)} modules successfully "
                        f"({len(names) - len(loaded)} failed)")
        if not {'translation_main', 'glossary_main'} & set(loaded):
            self.append_log("❌ Critical module loading failed - some functionality may be unavailable")
            return False
        return True

    def _attach_gui_logging_handlers(self):
        """Desktop: re-attach the log panel's logging handlers. Nothing to attach here."""
        return None

    def _create_watchdog_snapshot(self, context: str = None, model: str = None) -> None:
        """Desktop: snapshot for the API watchdog progress bar (mobile polls ``ProgressWatcher``)."""
        return None

    def _start_autoscroll_delay(self, ms=0):
        """Desktop: log panel auto-scroll delay."""
        return None

    def _update_manual_glossary_status(self):
        """Desktop: the main window's glossary status row."""
        return None

    def _record_library_raw_inputs(self, files):
        """Desktop: record the selected raw inputs in the Library's raw-inputs registry
        (``epub_library``, a Qt module). Skipped here: mobile imports inputs into Library/Raw
        through its FileBridge, and the run set-up must not import Qt."""
        return None


class GlossaryPipelineMixin(PipelineHooksMixin):
    """Glossary extraction and glossary auto-loading/auto-mapping, plus the input-selection
    helpers both pipelines use (moved verbatim; see the module docstring)."""

    def _glossary_editor_input_sources(self):
        """Return the current source identities used by the glossary editor.

        Extracted subtitle members from one ZIP collapse back to the archive
        path, so the editor uses the one archive-level glossary instead of
        treating every SRT/ASS/LRC member as a separate book.
        """
        try:
            files = list(getattr(self, 'selected_files', None) or [])
        except Exception:
            files = []

        fallback_path = None
        if not files:
            try:
                fallback_path = getattr(self, 'file_path', None)
            except Exception:
                fallback_path = None
        if not files and not fallback_path:
            try:
                getter = getattr(self, 'get_current_epub_path', None)
                fallback_path = getter() if callable(getter) else None
            except Exception:
                fallback_path = None

        from glossary_paths import resolve_glossary_input_sources

        return resolve_glossary_input_sources(
            files,
            fallback_path=fallback_path,
            subtitle_info_resolver=getattr(
                self,
                '_subtitle_zip_output_info',
                None,
            ),
        )

    def _is_special_file(self, filename):
        """Check if a filename is a special file using configurable keyword lists.
        
        Numbered HTML translation is handled separately; files whose name
        contains a digit can still be classified as special.
        """
        name_lower = filename.lower()
        base = os.path.basename(name_lower)
        # Strip response_ prefix (output filenames use this convention)
        if base.startswith("response_"):
            base = base[len("response_"):]
        name_noext = os.path.splitext(base)[0]
        # Check configured special-file patterns (substring match).
        # Non-numbered filenames are allowed to display as Ch.000 elsewhere, but that
        # must not make them special unless they match these configured lists.
        _kw_str = getattr(self, 'special_file_keywords_var', '')
        is_keyword_match = False
        if _kw_str:
            _keywords = [k.strip().lower() for k in _kw_str.split(',') if k.strip()]
            if any(kw in name_noext for kw in _keywords):
                is_keyword_match = True
        # Exact-match keywords
        if not is_keyword_match:
            _exact_str = getattr(self, 'special_file_exact_var', '')
            if _exact_str:
                _exact = [k.strip().lower() for k in _exact_str.split(',') if k.strip()]
                if name_noext in _exact:
                    is_keyword_match = True
        return is_keyword_match

    def _should_skip_special_file(self, filename, translate_special=False):
        """Return True when a configured special file should be skipped."""
        if translate_special:
            return False
        if not self._is_special_file(filename):
            return False
        if getattr(self, 'translate_all_numbered_html_var', True):
            import re as _re
            name = os.path.splitext(os.path.basename(str(filename or '')))[0]
            if name.lower().startswith("response_"):
                name = name[len("response_"):]
            if _re.search(r'\d', name):
                return False
        return True

    def _get_spine_filenames_for_preview(self, epub_path, start, end, spine_mode, translate_special=False):
        """
        Read the EPUB spine and return list of (position_label, filename, is_special_skipped) tuples
        that fall within the given range.
        If spine_mode is True, range refers to raw OPF spine position (1-based),
        matching the Progress Manager display.
        If spine_mode is False, range refers to chapter numbers extracted from filenames.
        is_special_skipped is True when translate_special is False and the file is a special file.
        """
        results = []
        try:
            import xml.etree.ElementTree as ET
            import zipfile

            with zipfile.ZipFile(epub_path, 'r') as zf:
                opf_path = find_epub_opf_member(zf)

                if not opf_path:
                    return results

                opf_content = zf.read(opf_path)
                root = ET.fromstring(opf_content)

                ns = {'opf': 'http://www.idpf.org/2007/opf'}
                if root.tag.startswith('{'):
                    default_ns = root.tag[1:root.tag.index('}')]
                    ns = {'opf': default_ns}

                # Build manifest
                manifest = {}
                for item in root.findall('.//opf:manifest/opf:item', ns):
                    item_id = item.get('id')
                    href = item.get('href')
                    media_type = item.get('media-type', '')
                    if item_id and href and (
                        'html' in media_type.lower() or
                        href.endswith(('.html', '.xhtml', '.htm'))
                    ):
                        manifest[item_id] = os.path.basename(href)

                # Get spine order
                spine = root.find('.//opf:spine', ns)
                spine_items = []
                if spine is not None:
                    for itemref in spine.findall('opf:itemref', ns):
                        idref = itemref.get('idref')
                        if idref and idref in manifest:
                            spine_items.append(manifest[idref])

                if spine_mode:
                    for i, fname in enumerate(spine_items):
                        raw_pos = i + 1
                        if not (start <= raw_pos <= end):
                            continue
                        skip_this = self._should_skip_special_file(fname, translate_special)
                        results.append((f"[{raw_pos:03d}]", fname, skip_this))
                else:
                    # Range refers to chapter numbers parsed from filenames
                    for i, fname in enumerate(spine_items):
                        is_special = self._is_special_file(fname)
                        skip_this = self._should_skip_special_file(fname, translate_special)

                        if is_special:
                            chap_num = 0
                        else:
                            matches = re.findall(r'(\d+)', fname)
                            if matches:
                                chap_num = int(matches[-1])
                            else:
                                chap_num = 0
                        if start <= chap_num <= end:
                            if skip_this:
                                results.append((f"Ch.{chap_num}", fname, True))
                            else:
                                results.append((f"Ch.{chap_num}", fname, False))

        except Exception as e:
            print(f"⚠️ Could not preview chapter range: {e}")

        return results

    def _get_opf_file_order(self, file_list):
        """
        Sort files based on OPF spine order if available.
        Uses STRICT OPF ordering - includes ALL files from spine without filtering.
        This ensures notice files, copyright pages, etc. are processed in the correct order.
        Returns sorted file list based on OPF, or original list if no OPF found.
        """
        source_files = list(file_list or [])

        # An EPUB/ZIP is the package container, not an item in another package's
        # spine.  Looking for an OPF below the EPUB's parent directory can pick
        # up an unrelated extracted book, then incorrectly compare outer
        # ``*.epub`` names with that book's internal ``*.html`` spine entries.
        # Preserve the user's batch order here; each EPUB is opened later and
        # its own internal OPF controls its chapter order.
        # Standalone HTML selections are prepared as individual EPUBs too.
        if any(
            os.path.splitext(os.fspath(path))[1].casefold() in {'.epub', '.zip', '.cbz', '.html', '.htm', '.xhtml'}
            for path in source_files
        ):
            return source_files

        try:
            import xml.etree.ElementTree as ET
            import zipfile
            import re
            
            # First, check for the workspace's authoritative OPF package.
            opf_file = None
            if file_list:
                current_dir = os.path.dirname(file_list[0]) if file_list else os.getcwd()
                opf_file = find_opf_path(current_dir)
                if opf_file:
                    self.append_log(f"📋 Found OPF package: {os.path.basename(opf_file)}")
            
            # If no OPF, check if any of the files is an OPF
            if not opf_file:
                for file_path in file_list:
                    if file_path.lower().endswith('.opf'):
                        opf_file = file_path
                        self.append_log(f"📋 Found OPF file: {os.path.basename(opf_file)}")
                        break
            
            # If no OPF, try to extract from EPUB
            if not opf_file:
                epub_files = [f for f in file_list if f.lower().endswith('.epub')]
                if epub_files:
                    epub_path = epub_files[0]
                    try:
                        with zipfile.ZipFile(epub_path, 'r') as zf:
                            opf_member = find_epub_opf_member(zf)
                            if opf_member:
                                opf_content = zf.read(opf_member)
                                temp_opf = os.path.join(os.path.dirname(epub_path), 'temp_content.opf')
                                with open(temp_opf, 'wb') as f:
                                    f.write(opf_content)
                                opf_file = temp_opf
                                self.append_log(f"📋 Extracted OPF from EPUB: {os.path.basename(epub_path)}")
                    except Exception as e:
                        self.append_log(f"⚠️ Could not extract OPF from EPUB: {e}")
            
            if not opf_file:
                self.append_log(f"ℹ️ No OPF file found, using default file order")
                return file_list
            
            # Parse the OPF file
            try:
                tree = ET.parse(opf_file)
                root = tree.getroot()
                
                # Handle namespaces
                ns = {'opf': 'http://www.idpf.org/2007/opf'}
                if root.tag.startswith('{'):
                    default_ns = root.tag[1:root.tag.index('}')]
                    ns = {'opf': default_ns}
                
                # Get manifest to map IDs to files
                manifest = {}
                for item in root.findall('.//opf:manifest/opf:item', ns):
                    item_id = item.get('id')
                    href = item.get('href')
                    
                    if item_id and href:
                        filename = os.path.basename(href)
                        manifest[item_id] = filename
                        # Store multiple variations for matching
                        name_without_ext = os.path.splitext(filename)[0]
                        manifest[item_id + '_noext'] = name_without_ext
                        # Also store with response_ prefix for matching
                        manifest[item_id + '_response'] = f"response_{filename}"
                        manifest[item_id + '_response_noext'] = f"response_{name_without_ext}"
                
                # Get spine order - include ALL files first for correct indexing
                spine_order_full = []
                spine = root.find('.//opf:spine', ns)
                if spine is not None:
                    for itemref in spine.findall('opf:itemref', ns):
                        idref = itemref.get('idref')
                        if idref and idref in manifest:
                            spine_order_full.append(manifest[idref])
                
                # Now filter out special files for processing (unless override is enabled)
                translate_special = os.environ.get('TRANSLATE_SPECIAL_FILES', '0') == '1'
                
                spine_order = []
                for item in spine_order_full:
                    # Numbered special files may be translated, but they still
                    # remain special for chapter numbering/progress display.
                    if not self._should_skip_special_file(item, translate_special):
                        spine_order.append(item)
                
                self.append_log(f"📋 Found {len(spine_order_full)} items in OPF spine ({len(spine_order)} after filtering)")
                
                # Count file types
                notice_count = sum(1 for f in spine_order if 'notice' in f.lower())
                chapter_count = sum(1 for f in spine_order if 'chapter' in f.lower() and 'notice' not in f.lower())
                skipped_count = len(spine_order_full) - len(spine_order)
                
                if skipped_count > 0:
                    self.append_log(f"   • Skipped files (cover/nav/toc): {skipped_count}")
                if notice_count > 0:
                    self.append_log(f"   • Notice/Copyright files: {notice_count}")
                if chapter_count > 0:
                    self.append_log(f"   • Chapter files: {chapter_count}")
                
                # Show first few spine entries
                if spine_order:
                    self.append_log(f"   📖 Spine order preview:")
                    for i, entry in enumerate(spine_order[:5]):
                        self.append_log(f"      [{i}]: {entry}")
                    if len(spine_order) > 5:
                        self.append_log(f"      ... and {len(spine_order) - 5} more")
                
                # Map input files to spine positions
                ordered_files = []
                unordered_files = []
                
                for file_path in file_list:
                    basename = os.path.basename(file_path)
                    basename_noext = os.path.splitext(basename)[0]
                    
                    # Try to find this file in the spine
                    found_position = None
                    matched_spine_file = None
                    
                    # Direct exact match
                    if basename in spine_order:
                        found_position = spine_order.index(basename)
                        matched_spine_file = basename
                    # Match without extension
                    elif basename_noext in spine_order:
                        found_position = spine_order.index(basename_noext)
                        matched_spine_file = basename_noext
                    else:
                        # Try pattern matching for response_ files
                        for idx, spine_item in enumerate(spine_order):
                            spine_noext = os.path.splitext(spine_item)[0]
                            
                            # Check if this is a response_ file matching spine item
                            if basename.startswith('response_'):
                                # Remove response_ prefix and try to match
                                clean_name = basename[9:]  # Remove 'response_'
                                clean_noext = os.path.splitext(clean_name)[0]
                                
                                if clean_name == spine_item or clean_noext == spine_noext:
                                    found_position = idx
                                    matched_spine_file = spine_item
                                    break
                                
                                # Try matching by chapter number
                                spine_num = re.search(r'(\d+)', spine_item)
                                file_num = re.search(r'(\d+)', clean_name)
                                if spine_num and file_num and spine_num.group(1) == file_num.group(1):
                                    # Check if both are notice or both are chapter files
                                    both_notice = 'notice' in spine_item.lower() and 'notice' in clean_name.lower()
                                    both_chapter = 'chapter' in spine_item.lower() and 'chapter' in clean_name.lower()
                                    if both_notice or both_chapter:
                                        found_position = idx
                                        matched_spine_file = spine_item
                                        break
                            else:
                                # For non-response files, check if spine item is contained
                                if spine_noext in basename_noext:
                                    found_position = idx
                                    matched_spine_file = spine_item
                                    break
                                
                                # Number-based matching
                                spine_num = re.search(r'(\d+)', spine_item)
                                file_num = re.search(r'(\d+)', basename)
                                if spine_num and file_num and spine_num.group(1) == file_num.group(1):
                                    # Check file type match
                                    both_notice = 'notice' in spine_item.lower() and 'notice' in basename.lower()
                                    both_chapter = 'chapter' in spine_item.lower() and 'chapter' in basename.lower()
                                    if both_notice or both_chapter:
                                        found_position = idx
                                        matched_spine_file = spine_item
                                        break
                    
                    if found_position is not None:
                        ordered_files.append((found_position, file_path))
                        self.append_log(f"  ✓ Matched: {basename} → spine[{found_position}]: {matched_spine_file}")
                    else:
                        unordered_files.append(file_path)
                        self.append_log(f"  ⚠️ Not in spine: {basename}")
                
                # Sort by spine position
                ordered_files.sort(key=lambda x: x[0])
                final_order = [f for _, f in ordered_files]
                
                # Add unmapped files at the end
                if unordered_files:
                    self.append_log(f"📋 Adding {len(unordered_files)} unmapped files at the end")
                    final_order.extend(sorted(unordered_files))
                
                # Clean up temp OPF if created
                if opf_file and 'temp_content.opf' in opf_file and os.path.exists(opf_file):
                    try:
                        os.remove(opf_file)
                    except:
                        pass
                
                self.append_log(f"✅ Files sorted using STRICT OPF spine order")
                self.append_log(f"   • Total files: {len(final_order)}")
                self.append_log(f"   • Following exact spine sequence from OPF")
                
                return final_order if final_order else file_list
                
            except Exception as e:
                self.append_log(f"⚠️ Error parsing OPF file: {e}")
                if opf_file and 'temp_content.opf' in opf_file and os.path.exists(opf_file):
                    try:
                        os.remove(opf_file)
                    except:
                        pass
                return file_list
                
        except Exception as e:
            self.append_log(f"⚠️ Error in OPF sorting: {e}")
            return file_list

    def run_glossary_extraction_direct(self, force_balanced_request_merging=False):
        """Run glossary extraction directly - handles multiple files and different file types"""
        try:
            if not str(getattr(self, 'model_var', '') or '').strip():
                self.append_log("❌ Glossary extraction stopped: no model is selected.")
                return False

            # Glossary extraction runs before the translation environment is
            # exported, so pass the live API retry settings here too.
            os.environ['MAX_RETRIES'] = str(self._resolve_max_retries())
            os.environ['INDEFINITE_RATE_LIMIT_RETRY'] = '1' if getattr(
                self, 'indefinite_rate_limit_retry_var',
                self.config.get('indefinite_rate_limit_retry', False),
            ) else '0'
            os.environ['GLOSSARY_REQUIRE_COMPLETE_BEFORE_TRANSLATION'] = '1' if self._live_bool_setting(
                'glossary_require_complete_checkbox',
                'glossary_require_complete_before_translation_var',
                'glossary_require_complete_before_translation',
                False,
            ) else '0'

            # Re-attach GUI logging handlers FIRST to reclaim logs from standalone header translation
            try:
                self._attach_gui_logging_handlers()
            except Exception:
                pass
            
            # Restore print hijack if it was captured by manga translator
            # This ensures main GUI logs go to main GUI, not manga GUI
            try:
                import builtins
                # Check if print was hijacked by manga translator
                if hasattr(builtins, '_manga_log_callbacks') and builtins._manga_log_callbacks:
                    # Restore original print for main GUI
                    if hasattr(builtins, 'print') and hasattr(builtins.print, '__name__'):
                        if builtins.print.__name__ == 'manga_print':
                            # Print is hijacked, restore it
                            from manga_translator import MangaTranslator
                            if hasattr(MangaTranslator, '_original_print_backup'):
                                builtins.print = MangaTranslator._original_print_backup
                                # Also restore in unified_api_client
                                try:
                                    import sys
                                    import unified_api_client
                                    uc_module = sys.modules.get('unified_api_client')
                                    if uc_module:
                                        uc_module.__dict__['print'] = MangaTranslator._original_print_backup
                                except Exception:
                                    pass
            except Exception:
                pass
            
            self.append_log("🔄 Loading glossary modules...")
            if not self._lazy_load_modules():
                self.append_log("❌ Failed to load glossary modules")
                return
            
            glossary_main = self._backend_entry('glossary_main')
            if glossary_main is None:
                self.append_log("❌ Glossary extraction module is not available")
                return

            # Reset again after lazy imports have completed. The first reset is
            # issued by run_glossary_extraction_thread(), but in a frozen build
            # the provider modules may not exist yet at that point, so their
            # process-wide cancellation events cannot be cleared. Do not erase
            # a genuine Stop clicked while module loading was in progress.
            if (
                not self.stop_requested
                and os.environ.get('TRANSLATION_CANCELLED') != '1'
                and os.environ.get('GRACEFUL_STOP') != '1'
            ):
                try:
                    import extract_glossary_from_epub
                    extract_glossary_from_epub.set_stop_flag(False)
                except Exception:
                    pass
                try:
                    import unified_api_client
                    if hasattr(unified_api_client, 'set_stop_flag'):
                        unified_api_client.set_stop_flag(False)
                except Exception:
                    pass
                try:
                    stop_file = os.environ.get('GLOSSARY_STOP_FILE')
                    if stop_file and os.path.exists(stop_file):
                        os.remove(stop_file)
                except Exception:
                    pass

            # Ensure streaming flags are applied to glossary runtime (mirrors translation flow)
            _force_stream = bool(getattr(self, '_force_stream_all', False))
            try:
                stream_on = _force_stream or bool(getattr(self, 'enable_streaming_var', self.config.get('enable_streaming', False)))
                os.environ['ENABLE_STREAMING'] = '1' if stream_on else '0'
                self.append_log(f"🛰️ Streaming {'enabled' if stream_on else 'disabled'} (exported ENABLE_STREAMING)")
            except Exception:
                pass
            try:
                allow_batch_logs = _force_stream or bool(getattr(self, 'allow_batch_stream_logs_var', self.config.get('allow_batch_stream_logs', False)))
                os.environ['ALLOW_BATCH_STREAM_LOGS'] = '1' if allow_batch_logs else '0'
            except Exception:
                pass
            try:
                allow_authgpt_logs = _force_stream or bool(getattr(self, 'allow_authgpt_batch_stream_logs_var', self.config.get('allow_authgpt_batch_stream_logs', False)))
                os.environ['ALLOW_AUTHGPT_BATCH_STREAM_LOGS'] = '1' if allow_authgpt_logs else '0'
            except Exception:
                pass
            try:
                stream_thinking = _force_stream or bool(getattr(self, 'stream_thinking_logs_var', self.config.get('stream_thinking_logs', False)))
                os.environ['STREAM_THINKING_LOGS'] = '1' if stream_thinking else '0'
            except Exception:
                pass
            if _force_stream:
                self._apply_forced_streaming_environment()

            # Sync all thinking-related env vars so glossary extraction uses the same settings as translation
            try:
                # Gemini thinking
                os.environ['ENABLE_GEMINI_THINKING'] = "1" if self.enable_gemini_thinking_var else "0"
                os.environ['THINKING_BUDGET'] = self.thinking_budget_var if self.enable_gemini_thinking_var else '0'
                os.environ['GEMINI_THINKING_LEVEL'] = getattr(self, 'thinking_level_var', 'high')
                os.environ['GEMINI_SERVICE_TIER'] = getattr(self, 'gemini_service_tier_var', 'off')
                os.environ['FORCE_SERVICE_TIER_UNKNOWN_ROUTES'] = '1' if self.force_service_tier_unknown_routes_var else '0'
                # GPT/OpenRouter reasoning
                os.environ['ENABLE_GPT_THINKING'] = "1" if self.enable_gpt_thinking_var else "0"
                os.environ['GPT_REASONING_TOKENS'] = self.gpt_reasoning_tokens_var if self.enable_gpt_thinking_var else ''
                os.environ['GPT_EFFORT'] = self.gpt_effort_var
                os.environ['OPENROUTER_USE_REASONING_TOKENS'] = '1' if self.openrouter_use_reasoning_tokens_var else '0'
                os.environ['PASS_THINKING_TO_OPENAI_COMPATIBLE'] = '1' if getattr(self, 'pass_thinking_all_openai_var', False) else '0'
                # DeepSeek thinking
                os.environ['ENABLE_DEEPSEEK_THINKING'] = "1" if getattr(self, 'enable_deepseek_thinking_var', True) else "0"
                os.environ['DEEPSEEK_EFFORT'] = getattr(self, 'deepseek_effort_var', 'high')
                os.environ['DEEPSEEK_USE_RESPONSES_API'] = "1" if getattr(self, 'deepseek_use_responses_api_var', False) else "0"
                # Anthropic extended/adaptive thinking
                os.environ['ENABLE_ANTHROPIC_THINKING'] = "1" if getattr(self, 'enable_anthropic_thinking_var', False) else "0"
                os.environ['ANTHROPIC_THINKING_BUDGET'] = str(self.anthropic_thinking_budget_var) if getattr(self, 'enable_anthropic_thinking_var', False) else '0'
                os.environ['ANTHROPIC_FORCE_ADAPTIVE'] = "1" if getattr(self, 'anthropic_force_adaptive_var', False) else "0"
                os.environ['ANTHROPIC_EFFORT'] = getattr(self, 'anthropic_effort_var', 'medium')
                # Skip thinking for lightweight tasks
                os.environ['SKIP_BOOK_TITLE_THINKING'] = "1" if getattr(self, 'skip_book_title_thinking_var', True) else "0"
                os.environ['SKIP_METADATA_THINKING'] = "1" if getattr(self, 'skip_metadata_thinking_var', True) else "0"
                os.environ['SKIP_TOC_THINKING'] = "1" if getattr(self, 'skip_toc_thinking_var', False) else "0"
                os.environ['LIGHTWEIGHT_THINKING_LEVEL'] = str(getattr(self, 'lightweight_thinking_level_var', 1))
            except Exception:
                pass

            if getattr(self, '_input_output_run_active', False):
                self._apply_direct_text_runtime_environment()

            if (
                self._has_epub_conversion_inputs()
                and not getattr(self, '_zip_inputs_resolved_for_current_run', False)
            ):
                self.append_log("📦 Preparing archive/HTML input(s) for glossary extraction...")
                self._resolve_zip_inputs_for_translation()
                if self.stop_requested:
                    self.append_log("⏹️ Glossary extraction cancelled during input preparation")
                    return

            # Create Glossary folder
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
            save_glossary_in_output = bool(self.config.get('save_glossary_in_output', False))
            if override_dir:
                glossary_base_dir = os.path.join(override_dir, "Glossary")
            else:
                glossary_base_dir = "Glossary"
            # On macOS .app bundles, cwd can be '/' (read-only root).
            # Resolve relative paths against the first selected file's directory.
            if not os.path.isabs(glossary_base_dir) and self.selected_files:
                glossary_base_dir = os.path.join(os.path.dirname(os.path.abspath(self.selected_files[0])), glossary_base_dir)
            os.makedirs(glossary_base_dir, exist_ok=True)
            
            # ========== NEW: APPLY OPF-BASED SORTING ==========
            # Sort files based on OPF order if available
            original_file_count = len(self.selected_files)
            has_epub_sources = any(
                os.path.splitext(os.fspath(path))[1].casefold() == '.epub'
                for path in self.selected_files
            )
            self.selected_files = self._get_opf_file_order(self.selected_files)
            if has_epub_sources:
                self.append_log(
                    f"📚 Processing {original_file_count} source file(s) in "
                    "selection order for glossary extraction; each EPUB uses "
                    "its own internal OPF spine"
                )
            else:
                self.append_log(f"📚 Processing {original_file_count} files in reading order for glossary extraction")
            # ====================================================
            
            # Group files by type and folder
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            
            # Separate images and text files
            image_files = []
            text_files = []
            seen_subtitle_bundle_ids = set()
            
            # Track successful items for summary
            successful_items = []
            
            for file_path in self.selected_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in image_extensions:
                    image_files.append(file_path)
                elif ext in {
                    '.epub', '.txt', '.pdf', '.sdlxliff',
                    '.srt', '.ass', '.lrc',
                }:
                    glossary_source = file_path
                    if ext in {'.srt', '.ass', '.lrc'}:
                        subtitle_info = self._subtitle_zip_output_info(file_path)
                        if isinstance(subtitle_info, dict):
                            archive_path = str(
                                subtitle_info.get('archive_path') or ''
                            ).strip()
                            bundle_id = str(
                                subtitle_info.get('bundle_id')
                                or archive_path
                            ).strip()
                            normalized_bundle_id = os.path.normcase(
                                os.path.abspath(bundle_id)
                            ) if bundle_id else ''
                            if (
                                normalized_bundle_id
                                and normalized_bundle_id
                                in seen_subtitle_bundle_ids
                            ):
                                continue
                            if normalized_bundle_id:
                                seen_subtitle_bundle_ids.add(
                                    normalized_bundle_id
                                )
                            if archive_path and os.path.isfile(archive_path):
                                # One subtitle ZIP produces one shared glossary,
                                # with one glossary chapter per archive member.
                                glossary_source = os.path.abspath(archive_path)
                    text_files.append(glossary_source)
                else:
                    self.append_log(f"⚠️ Skipping unsupported file type: {ext}")
            
            # Group images by folder
            image_groups = {}
            for img_path in image_files:
                folder = os.path.dirname(img_path)
                if folder not in image_groups:
                    image_groups[folder] = []
                image_groups[folder].append(img_path)
            
            total_groups = len(image_groups) + len(text_files)
            current_group = 0
            successful = 0
            failed = 0
            
            # Process image groups (each folder gets one combined glossary)
            for folder, images in image_groups.items():
                if self.stop_requested:
                    break
                
                current_group += 1
                folder_name = os.path.basename(folder) if folder else "images"
                
                self.append_log(f"\n{'='*60}")
                self.append_log(f"📁 Processing image folder ({current_group}/{total_groups}): {folder_name}")
                self.append_log(f"   Found {len(images)} images")
                self.append_log(f"{'='*60}")
                
                # Process all images in this folder and extract glossary
                if self._process_image_folder_for_glossary(folder_name, images, glossary_base_dir):
                    successful += 1
                    # Use absolute path for log if override is set, otherwise relative
                    if override_dir:
                        display_path = os.path.join(glossary_base_dir, f"{folder_name}_glossary.json")
                    else:
                        display_path = f"Glossary/{folder_name}_glossary.json"
                    successful_items.append(f"📁 {display_path} (Images)")
                else:
                    failed += 1
            
            # Process text files individually
            # Enable async extraction for PDFs to prevent GUI freezing
            os.environ['USE_ASYNC_CHAPTER_EXTRACTION'] = '1'
            for text_file in text_files:
                if self.stop_requested:
                    break
                
                current_group += 1
                
                self.append_log(f"\n{'='*60}")
                self.append_log(f"📄 Processing file ({current_group}/{total_groups}): {os.path.basename(text_file)}")
                self.append_log(f"{'='*60}")
                
                # Set output path environment variable for the script
                base_name = os.path.splitext(os.path.basename(text_file))[0]
                
                # Determine shared glossary directory. Glossary extraction
                # state belongs in repo/exe Glossary or output-override Glossary.
                override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
                save_glossary_in_output = bool(self.config.get('save_glossary_in_output', False))
                if override_dir:
                    glossary_dir = os.path.join(os.path.abspath(override_dir), "Glossary")
                else:
                    glossary_dir = os.path.join(_get_app_dir(), "Glossary")
                os.makedirs(glossary_dir, exist_ok=True)
                try:
                    from glossary_paths import get_book_glossary_path
                    output_path = get_book_glossary_path(glossary_dir, base_name, f"{base_name}_glossary.json")
                except Exception:
                    output_path = os.path.join(glossary_dir, base_name, f"{base_name}_glossary.json")
                os.environ["OUTPUT_PATH"] = output_path
                os.environ["GLOSSARY_SHARED_DIR"] = glossary_dir
                os.environ["SAVE_GLOSSARY_IN_OUTPUT"] = "1" if save_glossary_in_output else "0"
                if save_glossary_in_output:
                    os.environ["GLOSSARY_OUTPUT_BACKUP_DIR"] = self._output_side_glossary_backup_dir_for_source(text_file)
                else:
                    os.environ.pop("GLOSSARY_OUTPUT_BACKUP_DIR", None)
                
                if self._extract_glossary_from_text_file(
                    text_file,
                    force_balanced_request_merging=force_balanced_request_merging,
                ):
                    successful += 1
                    successful_items.append(f"📄 {output_path}")
                else:
                    # If failed but we have a partial file (checked inside _extract...), add to success list with note
                    # We need to manually check if partial file exists since _extract returns False on stop
                    partial_path = os.path.splitext(output_path)[0] + '.csv'
                    if os.path.exists(partial_path):
                        successful_items.append(f"📄 {partial_path} (Partial)")
                    failed += 1
            
            # Final summary
            self.append_log(f"\n{'='*60}")
            self.append_log(f"📊 Glossary Extraction Summary:")
            
            # If successful count is 0 but we have items in successful_items, it means we had partial success
            # This happens when a file was stopped mid-process but some chapters were saved
            if successful == 0 and len(successful_items) > 0:
                self.append_log(f"   ⚠️ Partial Success ({len(successful_items)}):")
                for item in successful_items:
                    self.append_log(f"      - {item}")
            elif successful > 0:
                self.append_log(f"   ✅ Successful ({successful}):")
                for item in successful_items:
                    self.append_log(f"      - {item}")
            else:
                self.append_log(f"   ✅ Successful: 0")
            
            if failed > 0:
                self.append_log(f"   ❌ Failed: {failed}")
            
            if self.stop_requested:
                self.append_log(f"   🛑 Process stopped by user")
                if successful == 0 and failed > 0:
                     self.append_log(f"      (Incomplete files marked as failed)")
            
            self.append_log(f"   📁 Total processed: {total_groups}")
            self.append_log(f"{'='*60}")
            
        except Exception as e:
            self.append_log(f"❌ Glossary extraction setup error: {e}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
        
        finally:
            # Save stop state for callers (balanced/full mode auto-trigger checks this)
            if self.stop_requested:
                self._glossary_stop_was_requested = True
            self.stop_requested = False
            glossary_stop_flag = self._backend_entry('glossary_stop_flag')
            if glossary_stop_flag:
                glossary_stop_flag(False)
            
            # IMPORTANT: Also reset the module's internal stop flag
            try:
                import extract_glossary_from_epub
                extract_glossary_from_epub.set_stop_flag(False)
            except:
                pass
                
            self.glossary_thread = None
            if hasattr(self, 'glossary_future'):
                try:
                    self.glossary_future = None
                except Exception:
                    pass
            self.current_file_index = 0
            # Emit signal to update button (thread-safe)
            self._ui_request('thread_complete')

    def _process_image_folder_for_glossary(self, folder_name, image_files, output_dir=None):
        """Process all images from a folder and create a combined glossary with new format"""
        try:
            import hashlib
            from unified_api_client import UnifiedClient, UnifiedClientError
            
            # Default output dir if not provided
            if not output_dir:
                output_dir = "Glossary"
            # On macOS .app bundles, cwd can be '/' (read-only root).
            if not os.path.isabs(output_dir) and image_files:
                output_dir = os.path.join(os.path.dirname(os.path.abspath(image_files[0])), output_dir)
            
            # Initialize folder-specific progress manager for images
            self.glossary_progress_manager = self._init_image_glossary_progress_manager(folder_name, output_dir)
            
            all_glossary_entries = []
            processed = 0
            skipped = 0
            
            # Get API key and model
            api_key = self.api_key_entry.text().strip()
            model = self.model_var
            
            # Check if model needs API key (delegates to UnifiedClient's authoritative list)
            try:
                from unified_api_client import UnifiedClient as _UC
                model_needs_api_key = _UC._model_needs_api_key(model)
            except Exception:
                model_needs_api_key = bool(model)  # safe fallback
            
            if (model_needs_api_key and not api_key) or not model:
                self.append_log("❌ Error: API key and model required")
                return False
            
            if not self.manual_glossary_prompt:
                self.append_log("❌ Error: No glossary prompt configured")
                return False
            
            # Propagate custom endpoint settings so UnifiedClient routes Gemini /
            # OpenAI / Anthropic requests through the user's configured endpoint.
            try:
                os.environ['USE_GEMINI_OPENAI_ENDPOINT'] = '1' if getattr(self, 'use_gemini_openai_endpoint_var', False) else '0'
                os.environ['GEMINI_OPENAI_ENDPOINT'] = getattr(self, 'gemini_openai_endpoint_var', '') or 'generativelanguage.googleapis.com'
                os.environ['OVERRIDE_GEMMA_FOR_CUSTOM_ENDPOINT'] = '1' if getattr(self, 'override_gemma_for_custom_endpoint_var', True) else '0'
                os.environ['USE_CUSTOM_OPENAI_ENDPOINT'] = '1' if getattr(self, 'use_custom_openai_endpoint_var', False) else '0'
                os.environ['OPENAI_CUSTOM_BASE_URL'] = getattr(self, 'openai_base_url_var', '') or ''
                _use_img_edit = bool(getattr(self, 'use_custom_image_edit_endpoint_var', False))
                os.environ['USE_CUSTOM_IMAGE_EDIT_ENDPOINT'] = '1' if _use_img_edit else '0'
                os.environ['CUSTOM_IMAGE_EDIT_BASE_URL'] = (getattr(self, 'custom_image_edit_endpoint_var', '') or '') if _use_img_edit else ''
                os.environ['OPENAI_IMAGE_EDIT_BASE_URL'] = (getattr(self, 'custom_image_edit_endpoint_var', '') or '') if _use_img_edit else ''
                os.environ['CUSTOM_OPENAI_PREFIX_ROUTES'] = self._custom_prefix_routes_env_json()
                os.environ['OLLAMA_SETTINGS_JSON'] = self._ollama_settings_env_json()
                os.environ['OPENAI_TTS_ENDPOINT'] = getattr(self, 'openai_tts_endpoint_var', '') or (getattr(self, 'openai_base_url_var', '') if str(getattr(self, 'openai_base_url_var', '')).rstrip('/').endswith('/audio/speech') else '')
                os.environ['GROQ_API_URL'] = getattr(self, 'groq_base_url_var', '') or ''
                os.environ['FIREWORKS_API_URL'] = getattr(self, 'fireworks_base_url_var', '') or ''
                os.environ['FORCE_NATIVE_ANTHROPIC'] = '1' if getattr(self, 'force_native_anthropic_var', False) else '0'
                os.environ['ANTHROPIC_BASE_URL'] = getattr(self, 'anthropic_base_url_var', '') or ''
            except Exception:
                pass
            
            # Initialize API client
            try:
                client = UnifiedClient(model=model, api_key=api_key)
            except Exception as e:
                self.append_log(f"❌ Failed to initialize API client: {str(e)}")
                return False
            
            # Get temperature and other settings from glossary config
            temperature = float(self.config.get('manual_glossary_temperature', 0.1))
            # Use glossary-specific output token limit; fall back to global if -1
            glossary_token_cfg = self.config.get('glossary_max_output_tokens', -1)
            if str(glossary_token_cfg) == '-1':
                max_tokens = int(self.max_output_tokens) if hasattr(self, 'max_output_tokens') else 8192
            else:
                max_tokens = int(glossary_token_cfg)
            api_delay = float(self.delay_entry.text()) if hasattr(self, 'delay_entry') else 2.0
            
            self.append_log(f"🔧 Glossary extraction settings:")
            self.append_log(f"   Temperature: {temperature}")
            self.append_log(f"   Max tokens: {max_tokens} ({'glossary override' if str(glossary_token_cfg) != '-1' else 'global'})")
            self.append_log(f"   API delay: {api_delay}s")
            format_parts = ["type", "raw_name", "translated_name", "gender"]
            custom_fields_json = self.config.get('manual_custom_fields', '[]')
            try:
                custom_fields = json.loads(custom_fields_json) if isinstance(custom_fields_json, str) else custom_fields_json
                if custom_fields:
                    format_parts.extend(custom_fields)
            except:
                custom_fields = []
            self.append_log(f"   Format: Simple ({', '.join(format_parts)})")
            
            # Check honorifics filter toggle
            honorifics_disabled = self.config.get('glossary_disable_honorifics_filter', False)
            if honorifics_disabled:
                self.append_log(f"   Honorifics Filter: ❌ DISABLED")
            else:
                self.append_log(f"   Honorifics Filter: ✅ ENABLED")
            
            # Track timing for ETA calculation
            start_time = time.time()
            total_entries_extracted = 0
            
            # Set up thread-safe payload directory
            thread_name = threading.current_thread().name
            thread_id = threading.current_thread().ident
            thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
            try:
                os.makedirs(thread_dir, exist_ok=True)
            except (PermissionError, OSError):
                import tempfile
                thread_dir = os.path.join(tempfile.gettempdir(), "Glossarion_Payloads", "glossary", f"{thread_name}_{thread_id}")
                try:
                    os.makedirs(thread_dir, exist_ok=True)
                except Exception:
                    pass
            
            # Process each image
            for i, image_path in enumerate(image_files):
                if self.stop_requested:
                    self.append_log("⏹️ Glossary extraction stopped by user")
                    break
                
                image_name = os.path.basename(image_path)
                self.append_log(f"\n   🖼️ Processing image {i+1}/{len(image_files)}: {image_name}")
                
                # Check progress tracking for this image
                try:
                    content_hash = self.glossary_progress_manager.get_content_hash(image_path)
                except Exception as e:
                    content_hash = hashlib.sha256(image_path.encode()).hexdigest()
                
                # Check if already processed
                needs_extraction, skip_reason, _ = self.glossary_progress_manager.check_image_status(image_path, content_hash)
                
                if not needs_extraction:
                    self.append_log(f"      ⏭️ {skip_reason}")
                    # Try to load previous results if available
                    existing_data = self.glossary_progress_manager.get_cached_result(content_hash)
                    if existing_data:
                        all_glossary_entries.extend(existing_data)
                    continue
                
                # Skip cover images
                if 'cover' in image_name.lower():
                    self.append_log(f"      ⏭️ Skipping cover image")
                    self.glossary_progress_manager.update(image_path, content_hash, status="skipped_cover")
                    skipped += 1
                    continue
                
                # Update progress to in-progress
                self.glossary_progress_manager.update(image_path, content_hash, status="in_progress")
                
                try:
                    # Read image
                    with open(image_path, 'rb') as img_file:
                        image_data = img_file.read()
                    
                    import base64
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    size_mb = len(image_data) / (1024 * 1024)
                    base_name = os.path.splitext(image_name)[0]
                    self.append_log(f"      📊 Image size: {size_mb:.2f} MB")
                    
                    # Build prompt for new format
                    custom_fields_json = self.config.get('manual_custom_fields', '[]')
                    try:
                        custom_fields = json.loads(custom_fields_json) if isinstance(custom_fields_json, str) else custom_fields_json
                    except:
                        custom_fields = []
                    
                    # Build honorifics instruction based on toggle
                    honorifics_instruction = ""
                    if not honorifics_disabled:
                        honorifics_instruction = "- Do NOT include honorifics (님, 씨, さん, 様, etc.) in raw_name\n"
                    
                    if self.manual_glossary_prompt:
                        prompt = self.manual_glossary_prompt
                        
                        # Build fields description
                        fields_str = """- type: "character" for people/beings or "term" for locations/objects/concepts
- raw_name: name in the original language/script  
- translated_name: English/romanized translation
- gender: (for characters only) Male/Female/Unknown"""
                        
                        if custom_fields:
                            for field in custom_fields:
                                fields_str += f"\n- {field}: custom field"
                        
                        # Build entries list from custom entry types (manual only)
                        def _entries_phrase(custom_types: dict) -> str:
                            items = []
                            for t_name, cfg in (custom_types or {}).items():
                                if cfg is not None and not cfg.get('enabled', True):
                                    continue
                                label = str(t_name).replace('_', ' ').strip()
                                if not label:
                                    continue
                                label = label[0].upper() + label[1:]
                                items.append(label)
                            if not items:
                                return "entries"
                            if len(items) == 1:
                                return f"{items[0]} entries"
                            if len(items) == 2:
                                return f"{items[0]} & {items[1]} entries"
                            return ", ".join(items[:-1]) + f", & {items[-1]} entries"

                        # Build fields1 description (\\x1F separated for CSV output)
                        header_parts = ['type', 'raw_name', 'translated_name', 'gender']
                        if custom_fields:
                            header_parts.extend(custom_fields)
                        sep_joined = '\\x1F'.join(header_parts)
                        fields1_str = f"Columns (separated by Unit Separator character \\x1F):\n{sep_joined}"
                        
                        entries_str = _entries_phrase(getattr(self, 'custom_entry_types', {}))
                        # Replace placeholders
                        prompt = prompt.replace('{fields1}', fields1_str)
                        prompt = prompt.replace('{{fields1}}', fields1_str)
                        prompt = prompt.replace('{fields}', fields_str)
                        prompt = prompt.replace('{{fields}}', fields_str)
                        prompt = prompt.replace('{entries}', entries_str)
                        prompt = prompt.replace('{{entries}}', entries_str)
                        prompt = prompt.replace('{chapter_text}', '')
                        prompt = prompt.replace('{{chapter_text}}', '')
                        prompt = prompt.replace('{text}', '')
                        prompt = prompt.replace('{{text}}', '')
                    else:
                        # Default prompt
                        fields_str = """For each entity, provide JSON with these fields:
- type: "character" for people/beings or "term" for locations/objects/concepts
- raw_name: name in the original language/script
- translated_name: English/romanized translation
- gender: (for characters only) Male/Female/Unknown"""
                        
                        if custom_fields:
                            fields_str += "\nAdditional custom fields:"
                            for field in custom_fields:
                                fields_str += f"\n- {field}"
                        
                        prompt = f"""Extract all characters and important terms from this image.

{fields_str}

Important rules:
{honorifics_instruction}- Romanize names appropriately
- Output ONLY a JSON array"""
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Save request payload in thread-safe location
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    payload_file = os.path.join(thread_dir, f"image_{timestamp}_{base_name}_request.json")
                    
                    request_payload = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model,
                        "image_file": image_name,
                        "image_size_mb": size_mb,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "processed_prompt": prompt,
                        "honorifics_filter_enabled": not honorifics_disabled
                    }
                    
                    with open(payload_file, 'w', encoding='utf-8') as f:
                        json.dump(request_payload, f, ensure_ascii=False, indent=2)
                    
                    self.append_log(f"      📝 Saved request: {os.path.basename(payload_file)}")
                    self.append_log(f"      🌐 Extracting glossary from image...")
                    
                    # API call with interrupt support (use 'glossary' context so empty responses are handled correctly)
                    response = self._call_api_with_interrupt(
                        client, messages, image_base64, temperature, max_tokens, context='glossary'
                    )
                    
                    # If the user requested stop *after* this API call returned:
                    # - Immediate stop: mark cancelled and exit
                    # - Graceful stop: keep and process this response, then stop before the next image
                    if self.stop_requested and not (bool(getattr(self, 'graceful_stop_active', False)) or (os.environ.get('GRACEFUL_STOP') == '1')):
                        self.append_log("⏹️ Glossary extraction stopped after API call")
                        self.glossary_progress_manager.update(image_path, content_hash, status="cancelled")
                        return False
                    
                    # Get response content
                    glossary_json = None
                    if isinstance(response, (list, tuple)) and len(response) >= 2:
                        glossary_json = response[0]
                    elif hasattr(response, 'content'):
                        glossary_json = response.content
                    elif isinstance(response, str):
                        glossary_json = response
                    else:
                        glossary_json = str(response)
                    
                    if glossary_json and glossary_json.strip():
                        # Save response in thread-safe location
                        response_file = os.path.join(thread_dir, f"image_{timestamp}_{base_name}_response.json")
                        response_payload = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "response_content": glossary_json,
                            "content_length": len(glossary_json)
                        }
                        with open(response_file, 'w', encoding='utf-8') as f:
                            json.dump(response_payload, f, ensure_ascii=False, indent=2)
                        
                        self.append_log(f"      📝 Saved response: {os.path.basename(response_file)}")
                        
                        # Parse the JSON response
                        try:
                            # Clean up the response
                            glossary_json = glossary_json.strip()
                            if glossary_json.startswith('```'):
                                glossary_json = glossary_json.split('```')[1]
                                if glossary_json.startswith('json'):
                                    glossary_json = glossary_json[4:]
                                glossary_json = glossary_json.strip()
                                if glossary_json.endswith('```'):
                                    glossary_json = glossary_json[:-3].strip()
                            
                            # Parse JSON
                            glossary_data = json.loads(glossary_json)
                            
                            # Process entries
                            entries_for_this_image = []
                            if isinstance(glossary_data, list):
                                for entry in glossary_data:
                                    # Validate entry format
                                    if isinstance(entry, dict) and 'type' in entry and 'raw_name' in entry:
                                        # Clean raw_name
                                        entry['raw_name'] = entry['raw_name'].strip()
                                        
                                        # Ensure required fields
                                        if 'translated_name' not in entry:
                                            entry['translated_name'] = entry.get('name', entry['raw_name'])
                                        
                                        # Add gender for characters if missing
                                        if entry['type'] == 'character' and 'gender' not in entry:
                                            entry['gender'] = 'Unknown'
                                        
                                        entries_for_this_image.append(entry)
                                        all_glossary_entries.append(entry)
                            
                            # Show progress
                            elapsed = time.time() - start_time
                            valid_count = len(entries_for_this_image)
                            
                            for j, entry in enumerate(entries_for_this_image):
                                total_entries_extracted += 1
                                
                                # Calculate ETA
                                if total_entries_extracted == 1:
                                    eta = 0.0
                                else:
                                    avg_time = elapsed / total_entries_extracted
                                    remaining_images = len(image_files) - (i + 1)
                                    estimated_remaining_entries = remaining_images * 3
                                    eta = avg_time * estimated_remaining_entries
                                
                                # Get entry name
                                entry_name = f"{entry['raw_name']} ({entry['translated_name']})"
                                
                                # Print progress
                                progress_msg = f'[Image {i+1}/{len(image_files)}] [{j+1}/{valid_count}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → {entry["type"]}: {entry_name}'
                                print(progress_msg)
                                self.append_log(progress_msg)
                            
                            self.append_log(f"      ✅ Extracted {valid_count} entries")
                            
                            # Update progress with extracted data
                            self.glossary_progress_manager.update(
                                image_path, 
                                content_hash, 
                                status="completed",
                                extracted_data=entries_for_this_image
                            )
                            
                            processed += 1
                            
                            # Save intermediate progress with skip logic
                            if all_glossary_entries:
                                self._save_intermediate_glossary_with_skip(folder_name, all_glossary_entries)
                            
                        except json.JSONDecodeError as e:
                            # Fallback: use the robust parser from extract_glossary_from_epub
                            # which handles JSON, CSV with headers, and headerless CSV
                            try:
                                from extract_glossary_from_epub import parse_api_response
                                parsed_entries = parse_api_response(glossary_json.strip())
                                entries_for_this_image = []
                                for entry in parsed_entries:
                                    if isinstance(entry, dict) and 'raw_name' in entry:
                                        entry['raw_name'] = entry['raw_name'].strip()
                                        if 'translated_name' not in entry:
                                            entry['translated_name'] = entry.get('name', entry['raw_name'])
                                        if entry.get('type') == 'character' and 'gender' not in entry:
                                            entry['gender'] = 'Unknown'
                                        entries_for_this_image.append(entry)
                                        all_glossary_entries.append(entry)
                                
                                if entries_for_this_image:
                                    self.append_log(f"      📋 Parsed {len(entries_for_this_image)} entries from CSV response")
                                    elapsed = time.time() - start_time
                                    for j, entry in enumerate(entries_for_this_image):
                                        total_entries_extracted += 1
                                        entry_name = f"{entry['raw_name']} ({entry.get('translated_name', '')})"
                                        progress_msg = f'[Image {i+1}/{len(image_files)}] [{j+1}/{len(entries_for_this_image)}] ({elapsed:.1f}s elapsed) → {entry.get("type", "?")}: {entry_name}'
                                        print(progress_msg)
                                        self.append_log(progress_msg)
                                    
                                    self.append_log(f"      ✅ Extracted {len(entries_for_this_image)} entries (fallback parser)")
                                    self.glossary_progress_manager.update(
                                        image_path, content_hash, status="completed",
                                        extracted_data=entries_for_this_image
                                    )
                                    processed += 1
                                    if all_glossary_entries:
                                        self._save_intermediate_glossary_with_skip(folder_name, all_glossary_entries)
                                else:
                                    raise ValueError("No valid entries found")
                            except Exception:
                                self.append_log(f"      ❌ Failed to parse response as JSON or CSV: {e}")
                                self.append_log(f"      Response preview: {glossary_json[:200]}...")
                                self.glossary_progress_manager.update(image_path, content_hash, status="error", error=str(e))
                                skipped += 1
                    else:
                        self.append_log(f"      ⚠️ No glossary data in response")
                        self.glossary_progress_manager.update(image_path, content_hash, status="error", error="No data")
                        skipped += 1
                    
                    # Add delay between API calls
                    if i < len(image_files) - 1 and not self.stop_requested:
                        self.append_log(f"      ⏱️ Waiting {api_delay}s before next image...")
                        elapsed = 0
                        while elapsed < api_delay and not self.stop_requested:
                            time.sleep(0.1)
                            elapsed += 0.1
                            
                except Exception as e:
                    self.append_log(f"      ❌ Failed to process: {str(e)}")
                    self.glossary_progress_manager.update(image_path, content_hash, status="error", error=str(e))
                    skipped += 1
            
            if not all_glossary_entries:
                self.append_log(f"❌ No glossary entries extracted from any images")
                return False
            
            self.append_log(f"\n📝 Extracted {len(all_glossary_entries)} total entries from {processed} images")
            
            # Save the final glossary with skip logic
            output_file = os.path.join(output_dir, f"{folder_name}_glossary.json")
            
            try:
                # Apply skip logic for duplicates
                self.append_log(f"📊 Applying skip logic for duplicate raw names...")
                
                # Import or define the skip function
                try:
                    from extract_glossary_from_epub import skip_duplicate_entries, remove_honorifics
                    # Set environment variable for honorifics toggle
                    os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if honorifics_disabled else '0'
                    final_entries = skip_duplicate_entries(
                        all_glossary_entries,
                        glossary_path=output_file,
                    )
                except:
                    # Fallback implementation
                    def remove_honorifics_local(name):
                        if not name or honorifics_disabled:
                            return name.strip()
                        
                        # Modern honorifics
                        korean_honorifics = ['님', '씨', '군', '양', '선생님', '사장님', '과장님', '대리님', '주임님', '이사님']
                        japanese_honorifics = ['さん', 'さま', '様', 'くん', '君', 'ちゃん', 'せんせい', '先生']
                        chinese_honorifics = ['先生', '女士', '小姐', '老师', '师傅', '大人']
                        
                        # Archaic honorifics
                        korean_archaic = ['공', '옹', '어른', '나리', '나으리', '대감', '영감', '마님', '마마']
                        japanese_archaic = ['どの', '殿', 'みこと', '命', '尊', 'ひめ', '姫']
                        chinese_archaic = ['公', '侯', '伯', '子', '男', '王', '君', '卿', '大夫']
                        
                        all_honorifics = (korean_honorifics + japanese_honorifics + chinese_honorifics + 
                                        korean_archaic + japanese_archaic + chinese_archaic)
                        
                        name_cleaned = name.strip()
                        sorted_honorifics = sorted(all_honorifics, key=len, reverse=True)
                        
                        for honorific in sorted_honorifics:
                            if name_cleaned.endswith(honorific):
                                name_cleaned = name_cleaned[:-len(honorific)].strip()
                                break
                        
                        return name_cleaned
                    
                    seen_raw_names = set()
                    final_entries = []
                    skipped = 0
                    
                    for entry in all_glossary_entries:
                        raw_name = entry.get('raw_name', '')
                        if not raw_name:
                            continue
                        
                        cleaned_name = remove_honorifics_local(raw_name)
                        
                        if cleaned_name.lower() in seen_raw_names:
                            skipped += 1
                            self.append_log(f"   ⏭️ Skipping duplicate: {raw_name}")
                            continue
                        
                        seen_raw_names.add(cleaned_name.lower())
                        final_entries.append(entry)
                    
                    self.append_log(f"✅ Kept {len(final_entries)} unique entries (skipped {skipped} duplicates)")
                
                # Save final glossary
                os.makedirs(output_dir, exist_ok=True)
                
                self.append_log(f"💾 Writing glossary to: {output_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_entries, f, ensure_ascii=False, indent=2)
                
                # Also save as CSV for compatibility
                csv_file = output_file.replace('.json', '.csv')
                with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    # Write header
                    header = ['type', 'raw_name', 'translated_name', 'gender']
                    if custom_fields:
                        header.extend(custom_fields)
                    writer.writerow(header)
                    
                    for entry in final_entries:
                        row = [
                            entry.get('type', ''),
                            entry.get('raw_name', ''),
                            entry.get('translated_name', ''),
                            entry.get('gender', '') if entry.get('type') == 'character' else ''
                        ]
                        # Add custom field values
                        if custom_fields:
                            for field in custom_fields:
                                row.append(entry.get(field, ''))
                        writer.writerow(row)
                
                self.append_log(f"💾 Also saved as CSV: {os.path.basename(csv_file)}")

                if bool(self.config.get('save_glossary_in_output', False)):
                    try:
                        import shutil
                        backup_root = os.environ.get('OUTPUT_DIRECTORY') or (
                            os.path.dirname(os.path.abspath(image_files[0])) if image_files else os.path.dirname(os.path.abspath(output_dir))
                        )
                        backup_dir = os.path.join(os.path.abspath(backup_root), "Glossary_Backup")
                        if os.path.normcase(os.path.abspath(backup_dir)) != os.path.normcase(os.path.abspath(output_dir)):
                            os.makedirs(backup_dir, exist_ok=True)
                            for src_path in (output_file, csv_file):
                                if src_path and os.path.exists(src_path):
                                    shutil.copy2(src_path, os.path.join(backup_dir, os.path.basename(src_path)))
                            self.append_log(f"💾 Also saved backup copies to: {backup_dir}")
                    except Exception as backup_err:
                        self.append_log(f"⚠️ Could not save glossary backup copies: {backup_err}")
                
                # Verify files were created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    self.append_log(f"✅ Glossary saved successfully ({file_size} bytes)")
                    
                    # Show sample of what was saved
                    if final_entries:
                        self.append_log(f"\n📋 Sample entries:")
                        for entry in final_entries[:5]:
                            self.append_log(f"   - [{entry['type']}] {entry['raw_name']} → {entry['translated_name']}")
                else:
                    self.append_log(f"❌ File was not created!")
                    return False
                
                return True
                
            except Exception as e:
                self.append_log(f"❌ Failed to save glossary: {e}")
                import traceback
                self.append_log(f"Full error: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            self.append_log(f"❌ Error processing image folder: {str(e)}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False

    def _init_image_glossary_progress_manager(self, folder_name, output_dir="Glossary"):
        """Initialize a folder-specific progress manager for image glossary extraction"""
        import hashlib
        
        class ImageGlossaryProgressManager:
            def __init__(self, folder_name, base_dir):
                self.PROGRESS_FILE = os.path.join(base_dir, f"{folder_name}_glossary_progress.json")
                self.prog = self._init_or_load()
            
            def _init_or_load(self):
                """Initialize or load progress tracking"""
                if os.path.exists(self.PROGRESS_FILE):
                    try:
                        with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                            return json.load(pf)
                    except Exception as e:
                        return {"images": {}, "content_hashes": {}, "extracted_data": {}, "version": "1.0"}
                else:
                    return {"images": {}, "content_hashes": {}, "extracted_data": {}, "version": "1.0"}
            
            def save(self):
                """Save progress to file atomically"""
                try:
                    import threading as _threading
                    import uuid as _uuid
                    os.makedirs(os.path.dirname(self.PROGRESS_FILE), exist_ok=True)
                    temp_file = (
                        f"{self.PROGRESS_FILE}."
                        f"{os.getpid()}."
                        f"{_threading.get_ident()}."
                        f"{_uuid.uuid4().hex}.tmp"
                    )
                    with open(temp_file, "w", encoding="utf-8") as pf:
                        json.dump(self.prog, pf, ensure_ascii=False, indent=2)
                    
                    os.replace(temp_file, self.PROGRESS_FILE)
                except Exception as e:
                    pass
            
            def get_content_hash(self, file_path):
                """Generate content hash for a file"""
                hasher = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()
            
            def check_image_status(self, image_path, content_hash):
                """Check if an image needs glossary extraction"""
                image_name = os.path.basename(image_path)
                
                # Check for skip markers
                skip_key = f"skip_{image_name}"
                if skip_key in self.prog:
                    skip_info = self.prog[skip_key]
                    if skip_info.get('status') == 'skipped':
                        return False, f"Image marked as skipped", None
                
                # Check if image has already been processed
                if content_hash in self.prog["images"]:
                    image_info = self.prog["images"][content_hash]
                    status = image_info.get("status")
                    
                    if status == "completed":
                        return False, f"Already processed", None
                    elif status == "skipped_cover":
                        return False, "Cover image - skipped", None
                    elif status == "error":
                        # Previous error, retry
                        return True, None, None
                
                return True, None, None
            
            def get_cached_result(self, content_hash):
                """Get cached extraction result for a content hash"""
                if content_hash in self.prog.get("extracted_data", {}):
                    return self.prog["extracted_data"][content_hash]
                return None
            
            def update(self, image_path, content_hash, status="in_progress", error=None, extracted_data=None):
                """Update progress for an image"""
                image_name = os.path.basename(image_path)
                
                image_info = {
                    "name": image_name,
                    "path": image_path,
                    "content_hash": content_hash,
                    "status": status,
                    "last_updated": time.time()
                }
                
                if error:
                    image_info["error"] = str(error)
                
                self.prog["images"][content_hash] = image_info
                
                # Store extracted data separately for reuse
                if extracted_data and status == "completed":
                    if "extracted_data" not in self.prog:
                        self.prog["extracted_data"] = {}
                    self.prog["extracted_data"][content_hash] = extracted_data
                
                self.save()
        
        # Create and return the progress manager
        progress_manager = ImageGlossaryProgressManager(folder_name, output_dir)
        self.append_log(f"📊 Progress tracking in: {output_dir}/{folder_name}_glossary_progress.json")
        return progress_manager

    def _save_intermediate_glossary_with_skip(self, folder_name, entries):
        """Save intermediate glossary results with skip logic"""
        try:
            # Determine output directory (same logic as main process)
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
            if override_dir:
                output_dir = os.path.join(override_dir, "Glossary")
            else:
                output_dir = "Glossary"
            # On macOS .app bundles, cwd can be '/' (read-only root).
            # Only on macOS — on Windows this changes the output dir and breaks glossary progress tracking.
            if sys.platform == 'darwin' and not os.path.isabs(output_dir) and hasattr(self, 'selected_files') and self.selected_files:
                output_dir = os.path.join(os.path.dirname(os.path.abspath(self.selected_files[0])), output_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{folder_name}_glossary.json")
            
            # Apply skip logic
            try:
                from extract_glossary_from_epub import skip_duplicate_entries
                unique_entries = skip_duplicate_entries(entries, glossary_path=output_file)
            except:
                # Fallback
                seen = set()
                unique_entries = []
                for entry in entries:
                    key = entry.get('raw_name', '').lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        unique_entries.append(entry)
            
            # Write the file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_entries, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.append_log(f"      ⚠️ Could not save intermediate glossary: {e}")

    def _call_api_with_interrupt(self, client, messages, image_base64, temperature, max_tokens, context='image_translation'):
        """Make API call with interrupt support and thread safety.

        IMPORTANT:
        - If graceful stop is active, we must NOT cancel an in-flight API call.
          We simply stop scheduling new work elsewhere.
        """
        import threading
        import queue
        from unified_api_client import UnifiedClientError
        
        result_queue = queue.Queue()
        
        def api_call():
            try:
                result = client.send_image(messages, image_base64, temperature=temperature, max_tokens=max_tokens, context=context)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))
        
        api_thread = threading.Thread(target=api_call)
        api_thread.daemon = True
        api_thread.start()
        
        # Check for stop every 0.5 seconds
        while api_thread.is_alive():
            if self.stop_requested:
                graceful = bool(getattr(self, 'graceful_stop_active', False)) or (os.environ.get('GRACEFUL_STOP') == '1')
                if not graceful:
                    # Immediate stop: cancel the operation
                    if hasattr(client, 'cancel_current_operation'):
                        client.cancel_current_operation()
                    raise UnifiedClientError("Glossary extraction stopped by user")
                # Graceful stop: do NOT cancel; just keep waiting for the API thread to finish.
            
            try:
                status, result = result_queue.get(timeout=0.5)
                if status == 'error':
                    raise result
                return result
            except queue.Empty:
                continue
        
        # Thread finished, get final result
        try:
            status, result = result_queue.get(timeout=1.0)
            if status == 'error':
                raise result
            return result
        except queue.Empty:
            raise UnifiedClientError("API call completed but no result received")

    def auto_load_glossary_for_file(self, file_path):
        """Automatically load a glossary from the EPUB's output folder.

        This is the *output-folder* auto-loader (e.g., `<output>/<book>/glossary.csv`).
        Returns True if a glossary was auto-loaded, otherwise False.
        """

        # CHECK FOR EPUB FIRST - before any clearing logic!
        if not file_path or not os.path.isfile(file_path):
            return False

        if not file_path.lower().endswith('.epub'):
            return False  # Exit early for non-EPUB files - don't touch glossaries!

        # Clear previous auto-loaded glossary if switching EPUB files
        if file_path != self.auto_loaded_glossary_for_file:
            # Only clear if the current glossary was auto-loaded AND not manually loaded
            if (
                self.auto_loaded_glossary_path and
                self.manual_glossary_path == self.auto_loaded_glossary_path and
                not getattr(self, 'manual_glossary_manually_loaded', False)
            ):
                self.manual_glossary_path = None
                self.append_log("📑 Cleared auto-loaded glossary from previous novel")

            self.auto_loaded_glossary_path = None
            self.auto_loaded_glossary_for_file = None

        # If the user manually loaded a glossary, keep using it until they clear it.
        if getattr(self, 'manual_glossary_manually_loaded', False) and self.manual_glossary_path:
            return False

        file_base = os.path.splitext(os.path.basename(file_path))[0]

        # Honor output directory override (matches translation output behavior)
        try:
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
        except Exception:
            override_dir = None

        if override_dir:
            output_dir = os.path.join(override_dir, file_base)
        else:
            output_dir = file_base

        # Prefer CSV over JSON when both exist
        glossary_candidates = [
            os.path.join(output_dir, "glossary.csv"),
            os.path.join(output_dir, "Glossary", "glossary.csv"),
            os.path.join(output_dir, "glossary.json"),
            os.path.join(output_dir, "Glossary", "glossary.json"),
            # TXT / MD support (lowest priority)
            os.path.join(output_dir, "glossary.txt"),
            os.path.join(output_dir, "Glossary", "glossary.txt"),
            os.path.join(output_dir, "glossary.md"),
            os.path.join(output_dir, "Glossary", "glossary.md"),
        ]

        for glossary_path in glossary_candidates:
            if not os.path.exists(glossary_path):
                continue

            ext = os.path.splitext(glossary_path)[1].lower()

            try:
                if ext == '.csv':
                    # Accept CSV without parsing
                    self.manual_glossary_path = glossary_path
                    self.auto_loaded_glossary_path = glossary_path
                    self.auto_loaded_glossary_for_file = file_path
                    self.manual_glossary_manually_loaded = False  # This is auto-loaded
                    _log_msg = f"📑 Auto-loaded glossary (output folder): {os.path.basename(glossary_path)}"
                    if getattr(self, '_last_glossary_log', '') != _log_msg:
                        self.append_log(_log_msg)
                        self._last_glossary_log = _log_msg
                    self._update_manual_glossary_status()
                    return True

                # TXT / MD: accept as-is
                if ext in ('.txt', '.md'):
                    self.manual_glossary_path = glossary_path
                    self.auto_loaded_glossary_path = glossary_path
                    self.auto_loaded_glossary_for_file = file_path
                    self.manual_glossary_manually_loaded = False
                    _log_msg = f"📑 Auto-loaded glossary (output folder): {os.path.basename(glossary_path)}"
                    if getattr(self, '_last_glossary_log', '') != _log_msg:
                        self.append_log(_log_msg)
                        self._last_glossary_log = _log_msg
                    self._update_manual_glossary_status()
                    return True

                # JSON: validate parse before accepting
                with open(glossary_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                self.manual_glossary_path = glossary_path
                self.auto_loaded_glossary_path = glossary_path
                self.auto_loaded_glossary_for_file = file_path
                self.manual_glossary_manually_loaded = False  # This is auto-loaded
                _log_msg = f"📑 Auto-loaded glossary (output folder): {os.path.basename(glossary_path)}"
                if getattr(self, '_last_glossary_log', '') != _log_msg:
                    self.append_log(_log_msg)
                    self._last_glossary_log = _log_msg
                self._update_manual_glossary_status()
                return True

            except Exception:
                # If parsing fails, try next candidate
                continue

        # Update restore button visibility for this file
        if hasattr(self, '_update_restore_visibility'):
            try:
                self._update_restore_visibility()
            except Exception:
                pass

        return False

    def _windows_supported_input_path(self, path):
        """Rename Windows-hostile input filenames before output-folder creation."""
        try:
            if not sys.platform.startswith('win'):
                return path
            if not path or path == "__generative_mode__" or not os.path.isfile(path):
                return path
            folder, filename = os.path.split(path)
            stem, ext = os.path.splitext(filename)
            safe_stem = stem.rstrip(" .")
            if safe_stem == stem:
                return path
            if not safe_stem:
                safe_stem = "input"

            candidate = os.path.join(folder, safe_stem + ext)
            if os.path.normcase(os.path.abspath(candidate)) == os.path.normcase(os.path.abspath(path)):
                return path

            counter = 2
            while os.path.exists(candidate):
                candidate = os.path.join(folder, f"{safe_stem}_windows_safe_{counter}{ext}")
                counter += 1

            os.rename(path, candidate)
            if ext.lower() == '.epub':
                try:
                    self._remap_windows_renamed_epub_glossary(path, candidate)
                except Exception:
                    pass
            try:
                self.append_log(
                    "⚠️ Windows does not support this filename for generated output folders because it ends in dots/spaces."
                )
                self.append_log(
                    f"📝 Renamed input file so extraction can continue: {filename} → {os.path.basename(candidate)}"
                )
            except Exception:
                pass
            return candidate
        except Exception as e:
            try:
                self.append_log(f"⚠️ Could not rename Windows-hostile input filename: {e}")
            except Exception:
                pass
            return path

    def _windows_glossary_rename_dirs(self):
        """Return shared Glossary dirs used by auto-mapping for source renames."""
        dirs = []

        try:
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
        except Exception:
            override_dir = None
        if override_dir:
            try:
                dirs.append(os.path.join(os.path.abspath(override_dir), 'Glossary'))
            except Exception:
                dirs.append(os.path.join(override_dir, 'Glossary'))

        try:
            base_dir = getattr(self, 'base_dir', '')
            if base_dir:
                dirs.append(os.path.join(base_dir, 'Glossary'))
        except Exception:
            pass

        try:
            dirs.append(os.path.join(_get_app_dir(), 'Glossary'))
        except Exception:
            pass

        unique = []
        seen = set()
        for path in dirs:
            if not path:
                continue
            try:
                key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
            except Exception:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    @staticmethod
    def _path_lookup_key(path):
        try:
            return os.path.normcase(os.path.normpath(os.path.abspath(path)))
        except Exception:
            return os.path.normcase(os.path.normpath(str(path or "")))

    def _remap_windows_renamed_epub_glossary(self, old_epub_path, new_epub_path):
        """Move generated glossary files and retarget state after an EPUB rename."""
        if not old_epub_path or not new_epub_path:
            return
        if not str(old_epub_path).lower().endswith('.epub') or not str(new_epub_path).lower().endswith('.epub'):
            return

        old_epub_abs = os.path.normpath(os.path.abspath(old_epub_path))
        new_epub_abs = os.path.normpath(os.path.abspath(new_epub_path))
        old_epub_key = self._path_lookup_key(old_epub_abs)
        moved_lookup = {}

        try:
            from glossary_paths import rename_auto_glossary_artifacts_for_book_rename

            rename_result = rename_auto_glossary_artifacts_for_book_rename(
                self._windows_glossary_rename_dirs(),
                os.path.splitext(os.path.basename(old_epub_path))[0],
                os.path.splitext(os.path.basename(new_epub_path))[0],
                logger=getattr(self, 'append_log', None),
            )
            for src, dst in rename_result.get('moved', []):
                moved_lookup[self._path_lookup_key(src)] = os.path.normpath(os.path.abspath(dst))

            moved_count = len(rename_result.get('moved', []))
            conflict_count = len(rename_result.get('conflicts', []))
            if moved_count:
                try:
                    self.append_log(f"📑 Renamed auto-mapped glossary files for Windows-safe EPUB name: {moved_count} file(s)")
                except Exception:
                    pass
            elif conflict_count:
                try:
                    self.append_log("⚠️ Kept existing glossary mapping because a renamed glossary destination already exists")
                except Exception:
                    pass
        except Exception as exc:
            try:
                self.append_log(f"⚠️ Could not rename associated glossary files: {exc}")
            except Exception:
                pass

        def _moved_glossary_path(path):
            if not path:
                return path, False
            replacement = moved_lookup.get(self._path_lookup_key(path))
            if replacement:
                return replacement, True
            return path, False

        changed = False

        try:
            manual_map = getattr(self, 'manual_glossary_map', None)
            if isinstance(manual_map, dict) and manual_map:
                updated_map = {}
                for key, glossary_path in manual_map.items():
                    try:
                        map_key_matches_epub = self._path_lookup_key(key) == old_epub_key
                    except Exception:
                        map_key_matches_epub = False

                    new_glossary_path, glossary_moved = _moved_glossary_path(glossary_path)
                    if map_key_matches_epub:
                        updated_map[new_epub_abs] = new_glossary_path
                        changed = True
                    else:
                        updated_map[key] = new_glossary_path
                        changed = changed or glossary_moved

                if updated_map != manual_map:
                    self.manual_glossary_map = updated_map
                    try:
                        self.config['manual_glossary_map'] = updated_map
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if self._path_lookup_key(getattr(self, 'auto_loaded_glossary_for_file', None)) == old_epub_key:
                self.auto_loaded_glossary_for_file = new_epub_abs
                changed = True
        except Exception:
            pass

        for attr in ('auto_loaded_glossary_path', 'manual_glossary_path'):
            try:
                current = getattr(self, attr, None)
                updated, did_move = _moved_glossary_path(current)
                if did_move and updated != current:
                    setattr(self, attr, updated)
                    changed = True
                    if attr == 'manual_glossary_path':
                        try:
                            self.config['manual_glossary_path'] = updated
                        except Exception:
                            pass
            except Exception:
                pass

        try:
            config_gp = self.config.get('manual_glossary_path')
            updated_config_gp, did_move = _moved_glossary_path(config_gp)
            if did_move and updated_config_gp != config_gp:
                self.config['manual_glossary_path'] = updated_config_gp
                changed = True
        except Exception:
            pass

        try:
            env_gp = os.environ.get('MANUAL_GLOSSARY')
            updated_env_gp, did_move = _moved_glossary_path(env_gp)
            if did_move and updated_env_gp:
                os.environ['MANUAL_GLOSSARY'] = updated_env_gp
                changed = True
            elif getattr(self, 'manual_glossary_path', None) and not getattr(self, 'manual_glossary_map', None):
                os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
        except Exception:
            pass

        if moved_lookup:
            try:
                cache = getattr(self, '_glossary_dir_candidate_cache', None)
                if isinstance(cache, dict):
                    cache.clear()
            except Exception:
                pass

        if changed:
            try:
                self._update_manual_glossary_status()
            except Exception:
                pass

    def _normalize_windows_input_filenames(self, paths):
        """Apply Windows filename normalization to a list of selected source paths."""
        changed = False
        normalized = []
        for path in list(paths or []):
            new_path = self._windows_supported_input_path(path)
            if new_path != path:
                changed = True
            normalized.append(new_path)

        if changed:
            try:
                if len(normalized) == 1 and hasattr(self, 'entry_epub'):
                    self.entry_epub.setText(normalized[0])
            except Exception:
                pass
            try:
                self.config['last_input_files'] = normalized
                source_files = [p for p in normalized if isinstance(p, str) and p.lower().endswith(('.epub', '.html', '.htm', '.xhtml', '.txt', '.pdf', '.md', '.sdlxliff', '.srt', '.ass', '.lrc'))]
                if source_files:
                    self.config['last_epub_path'] = source_files[0]
                self.save_config(show_message=False)
            except Exception:
                pass
        return normalized

    def _auto_load_glossary_after_extraction(self):
        """Auto-load the most recently generated glossary after extraction.
        
        Searches the Glossary/ folder for CSV/JSON files matching the current
        selected file(s) and sets manual_glossary_path + MANUAL_GLOSSARY env var.
        """
        try:
            files = list(getattr(self, 'selected_files', []) or [])
            if not files:
                return ""
            
            # Determine glossary base dir
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
            if override_dir:
                glossary_base_dir = os.path.join(
                    os.path.abspath(override_dir),
                    "Glossary",
                )
            else:
                # Glossary extraction writes text/subtitle glossaries beside
                # the application, not beside temporary ZIP members.
                glossary_base_dir = os.path.join(_get_app_dir(), "Glossary")
            
            if not os.path.isdir(glossary_base_dir):
                self.append_log(f"📑 No Glossary folder found after extraction")
                if not getattr(self, 'manual_glossary_manually_loaded', False):
                    self.manual_glossary_path = None
                    self.auto_loaded_glossary_path = None
                    self.auto_loaded_glossary_for_file = None
                    self.manual_glossary_map = {}
                    self.config['manual_glossary_path'] = ''
                    os.environ.pop('MANUAL_GLOSSARY', None)
                return ""
            
            # Build a set of candidate names to match against glossary filenames.
            # For text/EPUB files: use the file basename (e.g. "MyNovel")
            # For image files: also include the parent folder name since image
            # glossaries are named {folder_name}_glossary.json/csv
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            match_names = set()
            glossary_source_file = files[0]
            
            for file_path in files:
                base = os.path.splitext(os.path.basename(file_path))[0]
                ext = os.path.splitext(file_path)[1].lower()
                subtitle_archive_matched = False

                # Subtitle ZIP extraction replaces the selected archive with
                # temporary member paths. Include the original archive name so
                # the one shared archive-level glossary can be auto-loaded.
                if ext in {'.srt', '.ass', '.lrc'}:
                    subtitle_info = self._subtitle_zip_output_info(file_path)
                    if isinstance(subtitle_info, dict):
                        archive_path = str(
                            subtitle_info.get('archive_path') or ''
                        ).strip()
                        if archive_path:
                            subtitle_archive_matched = True
                            glossary_source_file = archive_path
                            match_names.add(
                                os.path.splitext(
                                    os.path.basename(archive_path)
                                )[0].casefold()
                            )

                if not subtitle_archive_matched:
                    match_names.add(base.casefold())

                # For images, also add the parent folder name.
                if ext in image_extensions:
                    folder = os.path.dirname(file_path)
                    folder_name = os.path.basename(folder) if folder else "images"
                    match_names.add(folder_name.casefold())
            try:
                from glossary_paths import migrate_all_legacy_glossary_files
                migrate_all_legacy_glossary_files(glossary_base_dir, logger=self.append_log)
            except Exception:
                pass
            
            # Look for glossary files matching any of the candidate names
            best_match = None
            best_mtime = 0
            
            try:
                for root, dirs, files_in_root in os.walk(glossary_base_dir):
                    if root != glossary_base_dir:
                        dirs[:] = []
                    for fn in files_in_root:
                        full = os.path.join(root, fn)
                        if not os.path.isfile(full):
                            continue
                        
                        stem, ext = os.path.splitext(fn)
                        ext_l = ext.lower()
                        if ext_l not in ('.csv', '.json'):
                            continue
                        
                        # Skip progress/metadata helpers
                        stem_lower = stem.lower()
                        if (
                            '_progress' in stem_lower
                            or stem_lower.endswith('_gender_tracker')
                            or stem_lower.endswith('_glossary_history')
                        ):
                            continue
                        
                        # Match exact source identities only. Substring matches
                        # can map short subtitle/member names to an unrelated
                        # book glossary.
                        stem_cf = stem.casefold()
                        parent_cf = os.path.basename(root).casefold()
                        for name_cf in match_names:
                            expected_stems = {
                                name_cf,
                                f"{name_cf}_glossary",
                            }
                            parent_match = (
                                parent_cf == name_cf
                                and stem_cf in {
                                    "glossary",
                                    name_cf,
                                    f"{name_cf}_glossary",
                                }
                            )
                            if stem_cf in expected_stems or parent_match:
                                mtime = os.path.getmtime(full)
                                # Prefer most recent, and CSV over JSON
                                priority = (1 if ext_l == '.csv' else 0)
                                if mtime > best_mtime or (mtime == best_mtime and priority > 0):
                                    best_match = full
                                    best_mtime = mtime
                                break
            except Exception:
                pass
            
            if best_match and os.path.exists(best_match):
                self.manual_glossary_path = best_match
                self.manual_glossary_manually_loaded = False
                self.auto_loaded_glossary_path = best_match
                self.auto_loaded_glossary_for_file = glossary_source_file
                # A stale multi-EPUB mapping must not suppress the one glossary
                # selected for the current subtitle archive.
                self.manual_glossary_map = {}
                self.config['manual_glossary_path'] = best_match
                os.environ['MANUAL_GLOSSARY'] = best_match
                self.append_log(f"📑 Auto-loaded generated glossary: {os.path.basename(best_match)}")
                return best_match  # Loaded successfully
            
            if not getattr(self, 'manual_glossary_manually_loaded', False):
                self.manual_glossary_path = None
                self.auto_loaded_glossary_path = None
                self.auto_loaded_glossary_for_file = None
                self.manual_glossary_map = {}
                self.config['manual_glossary_path'] = ''
                os.environ.pop('MANUAL_GLOSSARY', None)
            self.append_log(f"📑 No matching glossary found in {glossary_base_dir}")
            return ""
        except Exception as e:
            if not getattr(self, 'manual_glossary_manually_loaded', False):
                self.manual_glossary_path = None
                self.auto_loaded_glossary_path = None
                self.auto_loaded_glossary_for_file = None
                self.manual_glossary_map = {}
                try:
                    self.config['manual_glossary_path'] = ''
                except Exception:
                    pass
                os.environ.pop('MANUAL_GLOSSARY', None)
            self.append_log(f"⚠️ Failed to auto-load glossary: {e}")
            return ""

    def _autofill_glossary_for_current_selection(self) -> int:
        """Auto-fill glossary selection/mapping for the currently selected input files.

        Subtitle ZIP members are collapsed to their archive identity.
        A single source sets ``manual_glossary_path``; multiple sources use
        ``manual_glossary_map``.

        Returns the number of input sources that received an assignment.
        """
        try:
            sources = self._glossary_editor_input_sources()
        except Exception:
            sources = []
        if not sources:
            return 0

        # Single source: set one glossary path.
        if len(sources) == 1:
            try:
                if getattr(self, 'manual_glossary_path', None):
                    # If user manually loaded this glossary, don't override it
                    if getattr(self, 'manual_glossary_manually_loaded', False):
                        return 0
                    # Check what auto-mapping would assign.
                    # Only clear+re-map if there's a DIFFERENT glossary to replace with.
                    # If guess is None (no candidate) or same file, leave the current glossary alone.
                    try:
                        _peek = self._guess_glossary_for_input_file(sources[0])
                    except Exception:
                        _peek = None
                    if not _peek:
                        return 0  # No auto-mapping candidate — don't clear what's already loaded
                    if os.path.normpath(os.path.abspath(_peek)) == os.path.normpath(os.path.abspath(self.manual_glossary_path)):
                        return 1  # Already mapped to the right glossary
                    # Auto-mapping found a different glossary — clear the old one
                    _prev = self.manual_glossary_path
                    self.manual_glossary_path = None
                    self.auto_loaded_glossary_path = None
                    self.auto_loaded_glossary_for_file = None
            except Exception:
                pass

            gp = None
            try:
                gp = self._guess_glossary_for_input_file(sources[0])
            except Exception:
                gp = None

            if gp and os.path.exists(gp):
                try:
                    self.manual_glossary_path = gp
                    # This is auto-filled, not explicitly loaded by user
                    self.manual_glossary_manually_loaded = False
                    self.config['manual_glossary_path'] = gp
                except Exception:
                    pass
                try:
                    os.environ['MANUAL_GLOSSARY'] = gp
                except Exception:
                    pass
                try:
                    self.auto_loaded_glossary_path = gp
                except Exception:
                    pass
                try:
                    if hasattr(self, 'append_log'):
                        # This is Auto-Fill filename matching (not output-folder auto-load)
                        # Include fuzzy match percentage when present
                        _fuzzy_pct = getattr(self, '_last_fuzzy_match_pct', None)
                        self._last_fuzzy_match_pct = None  # consume it
                        _gp_base = os.path.basename(gp)
                        if _fuzzy_pct:
                            _log_msg = f"📑 Auto-mapped glossary (fuzzy {_fuzzy_pct}%): {_gp_base}"
                        else:
                            _log_msg = f"📑 Auto-mapped glossary: {_gp_base}"
                        # Deduplicate by glossary file basename (ignore fuzzy tag differences)
                        if getattr(self, '_last_automap_log_file', '') != _gp_base:
                            self.append_log(_log_msg)
                            self._last_automap_log_file = _gp_base
                except Exception:
                    pass
                self._update_manual_glossary_status()
                return 1

            return 0

        # Multiple inputs: build a per-source mapping.
        try:
            existing = getattr(self, 'manual_glossary_map', {}) or {}
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}

        mapping = {}
        assigned = 0
        log_pairs = []

        for source_path in sources:
            key = os.path.normpath(os.path.abspath(source_path))

            # Keep existing if valid
            try:
                prev = existing.get(source_path) or existing.get(key) or existing.get(os.path.normpath(source_path))
            except Exception:
                prev = None

            if prev and os.path.exists(prev):
                prev_n = os.path.normpath(os.path.abspath(prev))
                mapping[key] = prev_n
                assigned += 1
                try:
                    log_pairs.append((os.path.basename(source_path), os.path.basename(prev_n)))
                except Exception:
                    pass
                continue

            gp = None
            try:
                gp = self._guess_glossary_for_input_file(source_path)
            except Exception:
                gp = None

            if gp and os.path.exists(gp):
                gp_n = os.path.normpath(os.path.abspath(gp))
                mapping[key] = gp_n
                assigned += 1
                try:
                    log_pairs.append((os.path.basename(source_path), os.path.basename(gp_n)))
                except Exception:
                    pass

        try:
            # Enable mapping + disable global glossary path so it doesn't apply to all files.
            self.manual_glossary_map = mapping
            self.config['manual_glossary_map'] = mapping
            self.manual_glossary_path = None
            self.config['manual_glossary_path'] = ''
            self.manual_glossary_manually_loaded = False
        except Exception:
            pass

        try:
            if hasattr(self, 'append_log'):
                if assigned:
                    # Deduplicate: only log when the mapping actually changed
                    _map_key = tuple(sorted(mapping.items()))
                    if getattr(self, '_last_automap_multilog_key', None) != _map_key:
                        self._last_automap_multilog_key = _map_key
                        self._last_automap_no_match_logged = False  # Reset so future "no matches" can log once
                        self.append_log(
                            f"📑 Auto-mapped glossaries for "
                            f"{assigned}/{len(sources)} input source(s)"
                        )
                        # Show a short preview of mappings
                        try:
                            preview = log_pairs[:5]
                            for epub_name, gloss_name in preview:
                                self.append_log(f"   • {epub_name} → {gloss_name}")
                            if len(log_pairs) > 5:
                                self.append_log(f"   • …and {len(log_pairs) - 5} more")
                        except Exception:
                            pass
                else:
                    # Deduplicate: only log "no matches" once until a successful map resets the flag
                    if not getattr(self, '_last_automap_no_match_logged', False):
                        self.append_log("📑 Auto-map glossaries: no matches found")
                        self._last_automap_no_match_logged = True
        except Exception:
            pass

        return int(assigned)

    def _glossary_dir_signature(self, glossary_dir: str):
        """Return a shallow change signature for a Glossary directory."""
        try:
            root = os.path.abspath(glossary_dir)
            st = os.stat(root)
            child_dirs = []
            try:
                with os.scandir(root) as entries:
                    for entry in entries:
                        try:
                            if not entry.is_dir(follow_symlinks=False):
                                continue
                            entry_st = entry.stat(follow_symlinks=False)
                            child_dirs.append((entry.name.casefold(), entry_st.st_mtime_ns, entry_st.st_size))
                        except Exception:
                            child_dirs.append((getattr(entry, "name", "").casefold(), None, None))
            except Exception:
                child_dirs = []
            child_dirs.sort()
            return (os.path.normcase(root), st.st_mtime_ns, st.st_size, tuple(child_dirs))
        except Exception:
            return None

    def _get_glossary_dir_candidates(self, glossary_dir: str, ext_priority):
        """Return cached glossary file candidates for auto-mapping.

        The periodic auto-map timer calls this path on the GUI thread. Keep the
        observable matching behavior the same, but avoid re-walking unchanged
        Glossary folders every two seconds.
        """
        if not glossary_dir or not os.path.isdir(glossary_dir):
            return []

        try:
            glossary_dir = os.path.abspath(glossary_dir)
            cache_key = os.path.normcase(glossary_dir)
        except Exception:
            cache_key = glossary_dir

        ext_priority = tuple(ext_priority)
        signature = self._glossary_dir_signature(glossary_dir)
        try:
            cache = getattr(self, '_glossary_dir_candidate_cache', None)
            if not isinstance(cache, dict):
                cache = {}
                self._glossary_dir_candidate_cache = cache
            cached = cache.get(cache_key)
            if (
                signature is not None
                and cached
                and cached.get('signature') == signature
                and cached.get('ext_priority') == ext_priority
            ):
                return list(cached.get('candidates') or [])
        except Exception:
            cache = {}

        try:
            from glossary_paths import migrate_all_legacy_glossary_files
            migrate_all_legacy_glossary_files(glossary_dir)
        except Exception:
            pass

        candidates = []
        try:
            for root, dirs, files_in_root in os.walk(glossary_dir):
                if root != glossary_dir:
                    dirs[:] = []
                for fn in files_in_root:
                    full = os.path.join(root, fn)
                    if not os.path.isfile(full):
                        continue

                    stem, ext = os.path.splitext(fn)
                    stem_cf = stem.casefold()
                    ext_l = ext.lower()
                    if ext_l not in ext_priority:
                        continue

                    # Skip progress/metadata helpers
                    if (
                        stem_cf.endswith('_glossary_progress')
                        or stem_cf.endswith('glossary_progress')
                        or '_progress' in stem_cf
                        or stem_cf.endswith('_gender_tracker')
                        or stem_cf.endswith('_glossary_history')
                    ):
                        continue

                    candidates.append((stem_cf, ext_priority.index(ext_l), full))
        except Exception:
            return []

        try:
            signature = self._glossary_dir_signature(glossary_dir) or signature
            if signature is not None:
                cache[cache_key] = {
                    'signature': signature,
                    'ext_priority': ext_priority,
                    'candidates': list(candidates),
                }
                if len(cache) > 12:
                    for key in list(cache.keys())[:-12]:
                        cache.pop(key, None)
        except Exception:
            pass

        return list(candidates)

    def _guess_glossary_for_input_file(self, input_path: str):
        """Auto-detect a glossary for an input file.

        Search order (first hit wins):
        1) Output override Glossary folder: `<output override>/Glossary/`
        2) App/repo Glossary folder next to the executable/script
        3) CWD Glossary folder

        Matching rules (case-insensitive) for steps 1-3:
        - Prefer exact stem match to `<epub_stem>_glossary` (ignoring extension)
        - Fallback: exact stem match to `<epub_stem>`
        - Ignore progress files like `*_glossary_progress.*`
        """
        try:
            # Clear stale fuzzy score from any previous call so exact matches
            # aren't incorrectly tagged as fuzzy.
            self._last_fuzzy_match_pct = None

            if not input_path:
                return None

            base = os.path.splitext(os.path.basename(input_path))[0]
            base_cf = base.casefold()
            preferred_stems = [f"{base}_glossary".casefold(), base_cf]

            # Prefer CSV over JSON, then TXT/MD.
            ext_priority = [".csv", ".json", ".txt", ".md"]

            # Check if fuzzy auto-mapping is enabled
            _fuzzy_enabled = os.environ.get('FUZZY_AUTO_MAPPING', '0') == '1'
            try:
                _fuzzy_threshold = int(os.environ.get('FUZZY_AUTO_MAPPING_THRESHOLD', '80')) / 100.0
            except (ValueError, TypeError):
                _fuzzy_threshold = 0.80

            def _find_in_dir(glossary_dir: str):
                if not glossary_dir or not os.path.isdir(glossary_dir):
                    return None

                direct_matches = []
                fuzzy_candidates = []

                try:
                    for stem_cf, ext_rank, full in self._get_glossary_dir_candidates(glossary_dir, ext_priority):
                        if stem_cf in preferred_stems:
                            direct_matches.append((preferred_stems.index(stem_cf), ext_rank, full))
                        elif _fuzzy_enabled:
                            fuzzy_candidates.append((stem_cf, ext_rank, full))
                except Exception:
                    return None

                if direct_matches:
                    direct_matches.sort(key=lambda t: (t[0], t[1]))
                    return direct_matches[0][2]

                # Fuzzy fallback: score candidates against base name
                if _fuzzy_enabled and fuzzy_candidates:
                    from difflib import SequenceMatcher
                    best_score = 0.0
                    best_path = None
                    best_ext_rank = 99
                    for cand_stem, cand_ext_rank, cand_path in fuzzy_candidates:
                        # Score against both "stem_glossary" and plain stem
                        score = max(
                            SequenceMatcher(None, base_cf, cand_stem).ratio(),
                            SequenceMatcher(None, f"{base}_glossary".casefold(), cand_stem).ratio(),
                        )
                        if score > best_score or (score == best_score and cand_ext_rank < best_ext_rank):
                            best_score = score
                            best_path = cand_path
                            best_ext_rank = cand_ext_rank
                    if best_score >= _fuzzy_threshold and best_path:
                        # Store fuzzy score so the caller can include it in its log line
                        try:
                            self._last_fuzzy_match_pct = int(best_score * 100)
                        except Exception:
                            pass
                        return best_path

                return None

            override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')

            if override_dir:
                override_shared_glossary = os.path.join(os.path.abspath(override_dir), 'Glossary')
                hit = _find_in_dir(override_shared_glossary)
                if hit:
                    return hit

            # 1) App folder Glossary/
            try:
                app_glossary_dir = os.path.join(getattr(self, 'base_dir', ''), 'Glossary')
                hit = _find_in_dir(app_glossary_dir)
                if hit:
                    return hit
            except Exception:
                pass

            # 2) CWD Glossary/
            try:
                cwd_glossary_dir = os.path.join(_get_app_dir(), 'Glossary')
                hit = _find_in_dir(cwd_glossary_dir)
                if hit:
                    return hit
            except Exception:
                pass

            return None
        except Exception:
            return None

    def _copy_glossary_to_output_folders(self, glossary_path, input_files=None):
        """Copy a loaded glossary into the output folder of every currently
        selected input file, using the same target naming convention as the
        translator (glossary.csv / glossary.md / glossary.json).
        
        Falls back to ``self.entry_epub.text()`` and uses the same output-folder
        creation logic as Retranslation_GUI.py (creating an empty
        ``translation_progress.json`` if none exists yet).
        """
        import shutil
        import json as _json
        if not glossary_path or not os.path.isfile(glossary_path):
            return 0
        
        # Determine target filename based on extension (mirrors TransateKRtoEN.py logic)
        ext = os.path.splitext(glossary_path)[1].lower()
        if ext in ('.csv', '.txt'):
            target_name = 'glossary.csv'
        elif ext == '.md':
            target_name = 'glossary.md'
        elif ext == '.json':
            target_name = 'glossary.json'
        else:
            target_name = 'glossary.csv'
        
        # Resolve input files. Prefer selected_files; otherwise fall back to the
        # path shown in entry_epub (same fallback Retranslation_GUI uses).
        if input_files is None:
            input_files = [p for p in (getattr(self, 'selected_files', []) or []) if p]
        else:
            input_files = [p for p in (input_files or []) if p]
        if not input_files:
            try:
                fallback = self.entry_epub.text().strip() if hasattr(self, 'entry_epub') else ''
            except Exception:
                fallback = ''
            if fallback and not fallback.startswith('No file selected') and 'files selected' not in fallback \
                    and os.path.isfile(fallback):
                input_files = [fallback]
        
        if not input_files:
            self.append_log("ℹ️ No input file detected — cannot determine output folder for glossary.")
            return 0
        
        copied = 0
        seen_dirs = set()
        for file_path in input_files:
            try:
                # RPG Maker .exe uses its own GTool_Translation folder; skip
                if file_path.lower().endswith('.exe'):
                    continue
                # Use the authoritative resolver so extracted subtitle members
                # from one ZIP share the archive-level output folder. The old
                # duplicated basename logic created one directory per member in
                # Manual Glossary Only mode.
                output_dir = self._resolve_translation_output_dir(file_path)
                
                # Avoid copying multiple times into the same destination
                abs_out = os.path.abspath(output_dir)
                if abs_out in seen_dirs:
                    continue
                seen_dirs.add(abs_out)
                
                # Use the same folder-creation flow as Retranslation_GUI:
                # create the directory and seed an empty translation_progress.json
                # if none exists yet so downstream tools see a real output folder.
                folder_was_created = not os.path.exists(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                progress_file_path = os.path.join(output_dir, "translation_progress.json")
                if not os.path.exists(progress_file_path):
                    try:
                        empty_prog = {"chapters": {}, "chapter_chunks": {}, "version": "2.1"}
                        with open(progress_file_path, 'w', encoding='utf-8') as _pf:
                            _json.dump(empty_prog, _pf, ensure_ascii=False, indent=2)
                    except Exception as _pe:
                        self.append_log(f"⚠️ Could not seed translation_progress.json in {output_dir}: {_pe}")
                if folder_was_created:
                    self.append_log(f"📁 Created output folder: {output_dir}")
                    # Flash the PM button green if available (same UX as Retranslation_GUI)
                    try:
                        if hasattr(self, '_flash_pm_button_green'):
                            self._flash_pm_button_green(output_dir)
                    except Exception:
                        pass
                
                target_path = os.path.join(output_dir, target_name)
                if os.path.abspath(glossary_path) == os.path.abspath(target_path):
                    self.append_log(f"📑 Glossary already in output: {target_path}")
                    continue
                
                shutil.copy2(glossary_path, target_path)
                self.append_log(f"📑 Copied glossary to output: {target_path}")
                copied += 1
            except Exception as e:
                self.append_log(f"⚠️ Failed to copy glossary into output for '{file_path}': {e}")
        
        if copied:
            self.append_log(f"✅ Glossary loaded into {copied} output folder(s)")

        return copied

    def _sync_automapped_glossaries_to_output(self):
        """Overwrite output-side glossaries with the current auto-mapped files.

        Auto-mapping's source of truth remains the shared per-book Glossary
        folder. This mirror runs after any Balanced/Full extraction and before
        translation, so an existing output-side ``glossary.csv`` cannot remain
        stale after the user clicks Run Translation.
        """
        auto_mapping_modes = {
            'off', 'off_fuzzy_automap', 'balanced', 'full', 'single_pass'
        }
        try:
            if self._current_auto_glossary_mode() not in auto_mapping_modes:
                return 0
        except Exception:
            return 0

        # An explicitly loaded glossary is manual, even if the Auto-Mapping
        # checkbox is still visible in the current mode. Do not replace it.
        if getattr(self, 'manual_glossary_manually_loaded', False):
            return 0

        try:
            self._autofill_glossary_for_current_selection()
            sources = self._glossary_editor_input_sources()
        except Exception as exc:
            self.append_log(f"⚠️ Could not refresh auto-mapped glossaries: {exc}")
            return 0

        mapping = getattr(self, 'manual_glossary_map', {}) or {}
        global_glossary = getattr(self, 'manual_glossary_path', None)
        copied = 0
        synced_destinations = set()

        for source_path in sources:
            try:
                source_key = os.path.normpath(os.path.abspath(source_path))
                glossary_path = (
                    mapping.get(source_path)
                    or mapping.get(source_key)
                    or mapping.get(os.path.normpath(source_path))
                    or (global_glossary if len(sources) == 1 else None)
                )
                if not glossary_path or not os.path.isfile(glossary_path):
                    continue

                output_dir = os.path.normcase(os.path.abspath(
                    self._resolve_translation_output_dir(source_path)
                ))
                if output_dir in synced_destinations:
                    continue
                synced_destinations.add(output_dir)
                copied += self._copy_glossary_to_output_folders(
                    glossary_path,
                    input_files=[source_path],
                )
            except Exception as exc:
                self.append_log(
                    f"⚠️ Could not sync auto-mapped glossary for "
                    f"'{os.path.basename(source_path)}': {exc}"
                )

        if copied:
            self.append_log(
                f"✅ Synced latest auto-mapped glossary to "
                f"{copied} output folder(s)"
            )
        return copied


class TranslationPipelineMixin(GlossaryPipelineMixin, ImageJobMixin, RpgMakerJobMixin):
    """The translation run (set-up, worker, run_translation_direct) and QA-failure /
    multipass refinement planning (moved verbatim; see the module docstring).

    The image / video, generative-only and RPG Maker runners ``run_translation_direct``
    dispatches to come from ``ImageJobMixin`` / ``RpgMakerJobMixin`` (U7)."""

    def _clear_automatic_glossary_for_non_epub_selection(self, input_paths):
        """Drop a stale auto-selected glossary when switching away from EPUB input."""
        paths = [str(path or "") for path in (input_paths or []) if path]
        if any(path.lower().endswith(".epub") for path in paths):
            return False

        glossary_path = str(
            getattr(self, "manual_glossary_path", "") or ""
        ).strip()
        if getattr(self, "manual_glossary_manually_loaded", False):
            return False

        stale_candidates = [
            glossary_path,
            str(getattr(self, "auto_loaded_glossary_path", "") or "").strip(),
            str(os.environ.get("MANUAL_GLOSSARY", "") or "").strip(),
        ]
        try:
            stale_candidates.append(
                str(self.config.get("manual_glossary_path", "") or "").strip()
            )
        except Exception:
            pass
        cleared_path = next(
            (candidate for candidate in stale_candidates if candidate),
            "",
        )
        had_stale_state = bool(cleared_path)

        self.manual_glossary_path = None
        self.manual_glossary_manually_loaded = False
        self.auto_loaded_glossary_path = None
        self.auto_loaded_glossary_for_file = None
        self._last_glossary_log = ""
        try:
            self.config["manual_glossary_path"] = ""
        except Exception:
            pass
        os.environ.pop("MANUAL_GLOSSARY", None)
        try:
            if not had_stale_state:
                return False
            self.append_log(
                "📑 Cleared automatically selected glossary for new "
                f"non-EPUB input: {os.path.basename(cleared_path)}"
            )
        except Exception:
            pass
        try:
            self._update_manual_glossary_status()
        except Exception:
            pass
        return had_stale_state

    def _flatten_translation_qa_issue_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                parts.append(self._flatten_translation_qa_issue_text(key))
                parts.append(self._flatten_translation_qa_issue_text(item))
            return " ".join(part for part in parts if part)
        if isinstance(value, (list, tuple, set)):
            return " ".join(
                part
                for part in (self._flatten_translation_qa_issue_text(item) for item in value)
                if part
            )
        return str(value)

    def _is_foreign_character_translation_qa_issue(self, issue) -> bool:
        text = self._flatten_translation_qa_issue_text(issue).strip()
        if not text:
            return False
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        return "_text_found_" in normalized and "_chars_" in normalized

    def _entry_has_foreign_character_qa_failure(self, entry) -> bool:
        seen = set()
        current = entry
        while isinstance(current, dict) and id(current) not in seen:
            seen.add(id(current))
            status = str(current.get("status", "") or "").strip().lower()
            issues = []
            for key in ("qa_issues_found", "qa_issues", "failure_reason", "error_message"):
                value = current.get(key)
                if isinstance(value, (list, tuple, set)):
                    issues.extend(value)
                elif value is not None:
                    issues.append(value)
            if status == "qa_failed" and any(
                self._is_foreign_character_translation_qa_issue(issue) for issue in issues
            ):
                return True
            # A completed parent whose chunk ledger holds QA-failed chunks
            # mirrors those findings as {chunk index: [issues]}.
            chunk_issues = current.get("chunk_qa_issues_found")
            if isinstance(chunk_issues, dict):
                for chunk_value in chunk_issues.values():
                    chunk_list = (
                        chunk_value
                        if isinstance(chunk_value, (list, tuple, set))
                        else [chunk_value]
                    )
                    if any(
                        self._is_foreign_character_translation_qa_issue(issue)
                        for issue in chunk_list
                    ):
                        return True
            current = current.get("previous_progress_entry")
        return False

    def _collect_translation_qa_failures(self, files=None, *, foreign_character_only=False):
        """Collect qa_failed chapters from translation_progress.json files."""
        files = files or getattr(self, 'selected_files', []) or []
        failures = []
        seen = set()

        for file_path in files:
            if not file_path or file_path == "__generative_mode__":
                continue
            try:
                progress_path = os.path.join(
                    self._resolve_translation_output_dir(file_path),
                    "translation_progress.json",
                )
                if not os.path.exists(progress_path):
                    continue
                with open(progress_path, "r", encoding="utf-8") as pf:
                    progress = json.load(pf)
            except Exception:
                continue

            chapters = progress.get("chapters", {})
            if not isinstance(chapters, dict):
                continue

            for chapter_key, chapter_info in chapters.items():
                if not isinstance(chapter_info, dict):
                    continue
                status = str(chapter_info.get("status", "")).lower()
                if foreign_character_only and not self._entry_has_foreign_character_qa_failure(chapter_info):
                    continue
                issues = chapter_info.get("qa_issues_found") or []
                if isinstance(issues, str):
                    issues = [issues]
                elif isinstance(issues, (tuple, set)):
                    issues = list(issues)
                elif not isinstance(issues, list):
                    issues = []
                else:
                    issues = list(issues)
                # Chunk-level QA failures live on a completed parent as a
                # mirror of the chunk ledger. They are refinement targets
                # too, not chapters for the main phase to retranslate.
                chunk_issue_map = chapter_info.get("chunk_qa_issues_found")
                has_chunk_failures = bool(chapter_info.get("has_chunk_qa_failures")) or (
                    isinstance(chunk_issue_map, dict) and any(chunk_issue_map.values())
                )
                if isinstance(chunk_issue_map, dict):
                    for chunk_index, chunk_issues in sorted(
                        chunk_issue_map.items(), key=lambda item: str(item[0])
                    ):
                        if isinstance(chunk_issues, str):
                            chunk_issues = [chunk_issues]
                        for chunk_issue in chunk_issues or []:
                            label = f"chunk {chunk_index}: {chunk_issue}"
                            if label not in issues:
                                issues.append(label)
                if (
                    not foreign_character_only
                    and status not in {"qa_failed", "failed"}
                    and not chapter_info.get("qa_issues")
                    and not has_chunk_failures
                ):
                    continue

                chapter_num = (
                    chapter_info.get("pdf_section_num")
                    or chapter_info.get("actual_num")
                    or chapter_info.get("chapter_num")
                    or chapter_info.get("raw_chapter_num")
                    or chapter_key
                )
                output_file = chapter_info.get("output_file") or chapter_info.get("chapter_file") or ""
                dedupe_key = (os.path.abspath(progress_path), str(chapter_num), output_file)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                failures.append({
                    "source": os.path.basename(file_path),
                    "source_path": os.path.abspath(file_path),
                    "progress_path": os.path.abspath(progress_path),
                    "progress_key": str(chapter_key),
                    "chapter": chapter_num,
                    "pdf_section_num": chapter_info.get("pdf_section_num"),
                    "output_file": output_file,
                    "original_basename": chapter_info.get("original_basename") or "",
                    "issues": [str(issue).strip() for issue in issues if str(issue).strip()] or ["UNKNOWN"],
                })

        return failures

    @staticmethod
    def _chapter_scope_filename_key(value):
        """Normalize source and response filenames for preview/progress matching."""
        basename = os.path.basename(str(value or '').replace('\\', '/')).casefold()
        if basename.startswith('response_'):
            basename = basename[len('response_'):]
        return os.path.splitext(basename)[0]

    def _filter_translation_qa_failures_to_current_range(self, failures):
        """Limit pre-run multipass QA targets to the live chapter-range preview."""
        failures = list(failures or [])
        range_text, parsed_range, spine_order = RunEnvMixin._live_chapter_range_settings(self)
        if not range_text or not parsed_range:
            return failures

        start, end = parsed_range
        translate_special = bool(getattr(self, 'translate_special_files_var', False))
        allowed_epub_names = {}

        for failure in failures:
            source_path = str(failure.get('source_path') or '').strip()
            source_key = os.path.normcase(os.path.abspath(source_path)) if source_path else ''
            if not source_path.lower().endswith('.epub') or not os.path.isfile(source_path):
                continue
            if source_key in allowed_epub_names:
                continue
            preview_rows = self._get_spine_filenames_for_preview(
                source_path,
                start,
                end,
                spine_order,
                translate_special,
            )
            allowed_epub_names[source_key] = {
                TranslationPipelineMixin._chapter_scope_filename_key(filename)
                for _label, filename, is_special_skipped in preview_rows
                if not is_special_skipped
            }

        scoped = []
        for failure in failures:
            source_path = str(failure.get('source_path') or '').strip()
            source_key = os.path.normcase(os.path.abspath(source_path)) if source_path else ''
            allowed_names = allowed_epub_names.get(source_key)
            if allowed_names is not None:
                candidate_names = {
                    TranslationPipelineMixin._chapter_scope_filename_key(failure.get(field))
                    for field in ('original_basename', 'output_file')
                    if failure.get(field)
                }
                if candidate_names & allowed_names:
                    scoped.append(failure)
                    continue
                # Spine positions are not chapter numbers. If an old progress
                # row has no usable source filename, excluding it is safer than
                # refining a file the preview did not select.
                if spine_order:
                    continue

            try:
                chapter_num = float(
                    failure.get('pdf_section_num')
                    or failure.get('chapter')
                )
            except (TypeError, ValueError):
                continue
            if start <= chapter_num <= end:
                scoped.append(failure)
        return scoped

    def _translation_qa_failure_key(self, failure):
        return (
            str(failure.get("source", "")),
            str(failure.get("chapter", "")),
            str(failure.get("output_file", "")),
        )

    @staticmethod
    def _qa_failure_matches_resolution_request(failure, request):
        if not isinstance(failure, dict) or not isinstance(request, dict):
            return False
        request_source = str(request.get('source_path') or '').strip()
        failure_source = str(failure.get('source_path') or '').strip()
        if request_source and failure_source and (
            os.path.normcase(os.path.abspath(request_source))
            != os.path.normcase(os.path.abspath(failure_source))
        ):
            return False
        request_progress = str(request.get('progress_path') or '').strip()
        failure_progress = str(failure.get('progress_path') or '').strip()
        if request_progress and failure_progress and (
            os.path.normcase(os.path.abspath(request_progress))
            != os.path.normcase(os.path.abspath(failure_progress))
        ):
            return False
        request_key = str(request.get('progress_key') or '').strip()
        failure_key = str(failure.get('progress_key') or '').strip()
        if request_key and failure_key:
            return request_key == failure_key
        request_output = os.path.basename(
            str(request.get('output_file') or '')
        ).casefold()
        failure_output = os.path.basename(
            str(failure.get('output_file') or '')
        ).casefold()
        if request_output and failure_output:
            return request_output == failure_output
        return str(request.get('actual_num')) == str(failure.get('chapter'))

    def _prepare_multipass_qa_refinement_run(
        self,
        multipass_enabled,
        multipass_refinement_mode,
        requested_target=None,
    ):
        self._translation_run_output_mode_override = None
        self._translation_run_is_multipass_qa_refinement = False
        self._translation_run_qa_refinement_mode = ""
        self._translation_run_targeted_qa_failures = []
        self._translation_run_skipped_qa_failures = []
        self._translation_run_followup_translation_after_refinement = False
        self._translation_run_forced_multipass_mode = None

        if not multipass_enabled or multipass_refinement_mode not in ("failed", "partial", "partial.b", "partial.b2"):
            return []
        if (
            not requested_target
            and self._get_output_mode() in ("refinement", "audio", "image", "video")
        ):
            return []

        all_failures = TranslationPipelineMixin._filter_translation_qa_failures_to_current_range(
            self,
            self._collect_translation_qa_failures()
        )
        targeted_failures = TranslationPipelineMixin._filter_translation_qa_failures_to_current_range(
            self,
            self._collect_translation_qa_failures(foreign_character_only=True)
        )
        if requested_target:
            targeted_failures = [
                failure for failure in targeted_failures
                if self._qa_failure_matches_resolution_request(
                    failure, requested_target
                )
            ]
        if not targeted_failures:
            return []

        targeted_keys = {self._translation_qa_failure_key(failure) for failure in targeted_failures}
        skipped_failures = [] if requested_target else [
            failure
            for failure in all_failures
            if self._translation_qa_failure_key(failure) not in targeted_keys
        ]
        self._translation_run_output_mode_override = "refinement"
        self._translation_run_is_multipass_qa_refinement = True
        self._translation_run_qa_refinement_mode = multipass_refinement_mode
        self._translation_run_forced_multipass_mode = (
            multipass_refinement_mode if requested_target else None
        )
        self._translation_run_targeted_qa_failures = targeted_failures
        self._translation_run_skipped_qa_failures = skipped_failures
        self._translation_run_followup_translation_after_refinement = bool(skipped_failures)
        os.environ['OUTPUT_MODE'] = 'refinement'
        os.environ['ENABLE_REFINEMENT_OUTPUT_MODE'] = '1'
        os.environ['ENABLE_AUDIO_OUTPUT_MODE'] = '0'
        os.environ['ENABLE_IMAGE_OUTPUT_MODE'] = '0'
        os.environ['ENABLE_VIDEO_OUTPUT_MODE'] = '0'
        if requested_target:
            os.environ['PARTIAL_B_TARGET_PROGRESS_KEY'] = str(
                requested_target.get('progress_key') or ''
            )
            os.environ['PARTIAL_B_TARGET_OUTPUT_FILE'] = str(
                requested_target.get('output_file') or ''
            )
            os.environ['PARTIAL_B_TARGET_ACTUAL_NUM'] = str(
                requested_target.get('actual_num')
                if requested_target.get('actual_num') is not None else ''
            )
        return targeted_failures

    def _clear_translation_run_overrides(self):
        self._translation_run_output_mode_override = None
        self._translation_run_is_multipass_qa_refinement = False
        self._translation_run_qa_refinement_mode = ""
        self._translation_run_targeted_qa_failures = []
        self._translation_run_skipped_qa_failures = []
        self._translation_run_followup_translation_after_refinement = False
        self._translation_run_forced_multipass_mode = None
        os.environ.pop('PARTIAL_B_TARGET_PROGRESS_KEY', None)
        os.environ.pop('PARTIAL_B_TARGET_OUTPUT_FILE', None)
        os.environ.pop('PARTIAL_B_TARGET_ACTUAL_NUM', None)
        try:
            mode = self._get_output_mode()
            os.environ['OUTPUT_MODE'] = mode
            os.environ['ENABLE_REFINEMENT_OUTPUT_MODE'] = '1' if mode == 'refinement' else '0'
            os.environ['ENABLE_AUDIO_OUTPUT_MODE'] = '1' if mode == 'audio' else '0'
            os.environ['ENABLE_IMAGE_OUTPUT_MODE'] = self._get_allowed_image_output_mode()
            os.environ['ENABLE_VIDEO_OUTPUT_MODE'] = self._get_allowed_video_output_mode()
        except Exception:
            pass
        if getattr(self, '_input_output_run_active', False):
            self._apply_direct_text_runtime_environment()

    def _format_chapter_list(self, chapters):
        def _sort_key(value):
            text = str(value)
            try:
                return (0, int(text))
            except Exception:
                return (1, text)
        return ", ".join(str(chapter) for chapter in sorted(chapters, key=_sort_key))

    def _log_translation_qa_failure_summary(self, phase="current"):
        failures = self._collect_translation_qa_failures()
        if not failures:
            return

        self.append_log("")
        self.append_log("⚠️ QA failure summary:")
        if phase == "start":
            self.append_log("   Existing failed chapters were found before this run starts:")
        else:
            self.append_log("   Failed chapters found at the end of this translation run:")

        grouped = {}
        for failure in failures:
            source = failure["source"]
            for issue in failure["issues"]:
                grouped.setdefault((source, issue), []).append(failure["chapter"])

        for (source, issue), chapters in grouped.items():
            chapter_label = "Chapters" if len(chapters) != 1 else "Chapter"
            self.append_log(f"   - {source} - {chapter_label} {self._format_chapter_list(chapters)}: {issue}")
        issue_set = {issue for failure in failures for issue in failure["issues"]}
        self.append_log("")
        self.append_log("   What these QA issues mean:")
        if "TRUNCATED" in issue_set:
            self.append_log(
                "   - TRUNCATED: the provider/server ended the response early. Increase the compression factor "
                "or reduce the output token limit, use a different model, or increase the auto-retry truncated value."
            )
        if "PROHIBITED_CONTENT" in issue_set or "PROHIBITED CONTENT" in issue_set:
            self.append_log(
                "   • PROHIBITED_CONTENT: the model/provider likely blocked the request because of safety or censorship. "
                "Use a different model/provider for that chapter."
            )
        if "SPLIT_FAILED" in issue_set:
            self.append_log(
                "   • SPLIT_FAILED: the AI ignored or mishandled the split-marker instructions, so the output could not "
                "be safely mapped back to the original split chapters."
            )

        if "API_ERROR" in issue_set:
            self.append_log(
                "   - API_ERROR: the provider/API request failed before a usable response was returned. "
                "Retry the chapter, check the API/provider logs, or switch model/provider if it repeats."
            )

        known = {"TRUNCATED", "PROHIBITED_CONTENT", "PROHIBITED CONTENT", "SPLIT_FAILED", "API_ERROR"}
        unknown = sorted(issue for issue in issue_set if issue not in known)
        if unknown:
            self.append_log(
                f"   • Other issue(s) ({', '.join(unknown)}): the exact cause is not known from the saved QA marker. "
                "Retry the chapter, check the surrounding logs, or switch model/provider if it repeats."
            )

    def _prepare_translation_run(self, files=None):
        """Set up one translation run; returns a RunRequest, or None when nothing may start.

        Desktop ``run_translation_thread`` between its Run-button preflight and the worker
        thread (moved verbatim): input check (generative-mode sentinel, glossary of another
        book cleared), Library raw-input registry, stop-flag/run-id reset, Resolve-QA /
        multipass refinement planning, backend stop flags, client cancellation, glossary
        stop file, start logs and the existing QA-failure summary. *files* (mobile) replaces
        ``selected_files`` first; the desktop passes nothing and keeps its selection.
        """
        if files is not None:
            self.selected_files = list(files)
        # Check if files are selected
        _model_name = str(getattr(self, 'model_var', '')).strip()
        _is_generative_model = self._model_is_image_gen(_model_name) or self._model_is_video_gen(_model_name) or self._is_generative_output_mode()

        if not hasattr(self, 'selected_files') or not self.selected_files:
            file_path = self.entry_epub.text().strip()
            if not file_path or file_path.startswith("No file selected") or "files selected" in file_path:
                if _is_generative_model:
                    # Image/video generation models don't need an input file;
                    # use a synthetic sentinel so the rest of the pipeline works
                    self.selected_files = ["__generative_mode__"]
                    self.append_log(
                        f"🎨 Generative model detected ({_model_name}) – "
                        "running without an input file."
                    )
                else:
                    self._ui_message('critical', "Error", "Please select file(s) to translate.")
                    return
            else:
                self.selected_files = [file_path]
                self.selected_files = self._normalize_windows_input_filenames(self.selected_files)
                file_path = self.selected_files[0]
            
            # Auto-clear glossary if file doesn't match (works for both manual and auto-loaded)
            if self.manual_glossary_path:
                current_file_base = os.path.splitext(os.path.basename(file_path))[0]
                # Check both glossary filename AND parent folder name
                glossary_full_path = self.manual_glossary_path
                glossary_name = os.path.basename(glossary_full_path)
                glossary_parent = os.path.basename(os.path.dirname(glossary_full_path))
                
                # Check if the current file's base name appears in the glossary path (filename or parent folder)
                if current_file_base not in glossary_name and current_file_base not in glossary_parent:
                    # Glossary doesn't match, clear it
                    old_glossary = glossary_full_path
                    was_manual = getattr(self, 'manual_glossary_manually_loaded', False)
                    source_type = "manually loaded" if was_manual else "auto-loaded"
                    self.append_log(f"📑 Cleared {source_type} glossary from different source: {os.path.basename(os.path.dirname(old_glossary))}")
                    self.manual_glossary_path = None
                    self.manual_glossary_manually_loaded = False
                    self.auto_loaded_glossary_path = None
                    self.auto_loaded_glossary_for_file = None
        else:
            self.selected_files = self._normalize_windows_input_filenames(self.selected_files)

        # Re-check at run time as well as selection time. This covers restored
        # sessions and paths edited before the worker starts.
        self._clear_automatic_glossary_for_non_epub_selection(
            self.selected_files
        )
        
        # Record every selected raw input in the Library's raw-inputs
        # registry so the library dialog can find it later (even if the
        # file itself lives outside the Library folder).
        if not getattr(self, '_input_output_run_active', False):
            self._record_library_raw_inputs(self.selected_files)

        # Reset stop flags
        self.stop_requested = False
        self._glossary_stop_was_requested = False  # Reset glossary stop flag from previous run
        self._translation_anti_duplicate_logged = False
        self.graceful_stop_active = False  # Reset graceful stop state
        self._zip_inputs_resolved_for_current_run = False
        reset_stop_env('translation')

        # Assign a new run id so transport logs (httpx) can be suppressed for stale previous runs
        os.environ['GLOSSARION_RUN_ID'] = make_run_id('translation')

        qa_resolution_request = getattr(
            self, '_single_qa_resolution_request', None
        )
        if isinstance(qa_resolution_request, dict):
            qa_resolution_request = dict(qa_resolution_request)
        else:
            qa_resolution_request = None
        self._clear_translation_run_overrides()
        try:
            # Multipass preflight must see the same live range and spine-order
            # selection that the preview and translation worker will use.
            self._export_chapter_range_runtime_env()
            if getattr(self, '_metadata_only_run', False):
                multipass_enabled = False
                multipass_refinement_mode = 'failed'
                os.environ['MULTIPASS_MODE'] = '0'
            elif qa_resolution_request:
                multipass_enabled = True
                multipass_refinement_mode = 'partial.b'
                os.environ['MULTIPASS_MODE'] = '1'
                os.environ['MULTIPASS_REFINEMENT_MODE'] = 'partial.b'
            else:
                multipass_enabled, multipass_refinement_mode = self._export_multipass_runtime_env()
            multipass_refinement_mode_label = (
                'Full + raw'
                if multipass_refinement_mode == 'full_with_raw'
                else multipass_refinement_mode.replace('_', ' ').title()
            )
            targeted_refinement_failures = self._prepare_multipass_qa_refinement_run(
                multipass_enabled,
                multipass_refinement_mode,
                requested_target=qa_resolution_request,
            )
            if qa_resolution_request and not targeted_refinement_failures:
                self.append_log(
                    "⚠️ Resolve QA issue cancelled: the selected entry no "
                    "longer has a raw foreign-text QA failure"
                )
                self._single_qa_resolution_request = None
                self._clear_translation_run_overrides()
                return
            if multipass_enabled:
                self.append_log(
                    f"Multipass refinement mode: {multipass_refinement_mode_label} "
                    "(exported for translation)"
                )
            if targeted_refinement_failures:
                chapters = [failure["chapter"] for failure in targeted_refinement_failures]
                if qa_resolution_request:
                    target_output = qa_resolution_request.get(
                        'output_file'
                    ) or self._format_chapter_list(chapters)
                    self.append_log(
                        "Partial.b: resolving only the selected raw "
                        f"foreign-text QA entry ({target_output})"
                    )
                else:
                    self.append_log(
                        f"{multipass_refinement_mode_label} multipass: existing foreign-character QA failures found; "
                        "running refinement instead of translation "
                        f"(chapters: {self._format_chapter_list(chapters)})"
                    )
                skipped_failures = getattr(self, '_translation_run_skipped_qa_failures', [])
                if skipped_failures:
                    skipped_chapters = [failure["chapter"] for failure in skipped_failures]
                    self.append_log(
                        "Other QA-failed entries will be retried through normal translation after refinement "
                        f"(chapters: {self._format_chapter_list(skipped_chapters)})"
                    )
        except Exception as e:
            self.append_log(f"Warning: Could not export multipass refinement mode: {e}")
            if qa_resolution_request:
                self.append_log(
                    "⚠️ Resolve QA issue cancelled before the targeted run started"
                )
                self._single_qa_resolution_request = None
                self._clear_translation_run_overrides()
                return

        translation_stop_flag = self._backend_entry('translation_stop_flag')
        if translation_stop_flag:
            translation_stop_flag(False)
        
        # Also reset the module's internal stop flag
        try:
            if hasattr(self, '_main_module') and self._main_module:
                if hasattr(self._main_module, 'set_stop_flag'):
                    self._main_module.set_stop_flag(False)
        except:
            pass

        # Close the previous run's lingering streams, then reset the client's
        # global cancellation (streaming stop) for the new run
        clear_client_cancellation()

        # Create a watchdog snapshot immediately when starting translation
        try:
            self._create_watchdog_snapshot(context="translation", model=getattr(self, 'model_var', None))
        except Exception:
            pass

        # Prepare shared stop-file for glossary/translation workers
        prepare_glossary_stop_file()
        
        # Update button immediately to show translation is starting
        if hasattr(self, 'button_run'):
            self.button_run.config(text="⏹ Stop", state="normal")
        
        # Delay auto-scroll so first log is readable (set to 0 for immediate scrolling)
        self._start_autoscroll_delay(0)
        # Show immediate feedback that translation is starting
        if getattr(self, '_translation_run_is_multipass_qa_refinement', False):
            mode_label = str(getattr(self, '_translation_run_qa_refinement_mode', '') or 'multipass').title()
            self.append_log(f"🚀 Initializing {mode_label} multipass refinement...")
        elif getattr(self, '_metadata_only_run', False):
            self.append_log("🌐 Initializing metadata translation...")
        else:
            self.append_log("🚀 Initializing translation process...")
        
        # Debug: Log stop behavior settings
        graceful_stop = getattr(self, 'graceful_stop_var', False)
        wait_for_chunks = getattr(self, 'wait_for_chunks_var', True)
        print(f"🔧 Stop settings: graceful_stop={graceful_stop}, wait_for_chunks={wait_for_chunks}")
        # Force immediate scroll to bottom so user sees the latest output right away
        try:
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass
        try:
            self._log_translation_qa_failure_summary(phase="start")
        except Exception as e:
            self.append_log(f"⚠️ Could not read existing QA failure summary: {e}")
        # Reset stop notice dedupe flag at start of a run
        self._stop_notice_shown = False

        return RunRequest(
            files=list(getattr(self, 'selected_files', None) or []),
            run_id=os.environ.get('GLOSSARION_RUN_ID', ''),
            qa_resolution_request=qa_resolution_request,
            multipass_qa_refinement=bool(getattr(self, '_translation_run_is_multipass_qa_refinement', False)),
            refinement_mode=str(getattr(self, '_translation_run_qa_refinement_mode', '') or ''),
            followup_translation=bool(getattr(self, '_translation_run_followup_translation_after_refinement', False)),
            metadata_only=bool(getattr(self, '_metadata_only_run', False)),
            single_chapter_filter=getattr(self, '_single_chapter_filter', None),
            direct_text=bool(getattr(self, '_input_output_run_active', False)),
            graceful_stop=bool(graceful_stop),
            wait_for_chunks=bool(wait_for_chunks),
        )

    def _translation_worker(self, request):
        """Run one prepared translation (desktop: run_translation_thread's worker thread).

        The body of the desktop ``simple_thread_target`` closure, moved verbatim: load the
        backend, prepare archive/HTML inputs, large-EPUB extraction settings, the
        Balanced/Full pre-translation glossary pass (failed-chapter retries, require-complete
        gate, Direct Text approval), ``run_translation_direct`` plus the multipass
        refinement follow-up, the post-translation QA trigger, and the end-of-run reset in
        ``finally``. The closure captured no locals of run_translation_thread (it reads the
        owner), so *request* (from ``_prepare_translation_run``) is not read here.

        Returns the run's outcome (body edit; the desktop thread ignores it): what
        ``run_translation_direct`` returned (True when at least one file translated), or
        False when the run ended early (modules not loaded, stopped while preparing inputs
        or during the glossary pass, glossary approval declined, the require-complete
        glossary gate, an error caught here). Mobile job adapters map False to Failed or,
        after a Stop, to Stopped.
        """
        try:
            self.append_log("🟢 Thread started successfully!")
            
            # Load modules if needed
            if not self._modules_loaded:
                self.append_log("📬 Loading translation modules...")
                if not self._lazy_load_modules():
                    self.append_log("❌ Failed to load modules")
                    return False
                self.append_log("✅ Modules loaded")

            if self._has_epub_conversion_inputs():
                self.append_log("📦 Preparing archive/HTML input(s) in translation worker...")
                self._resolve_zip_inputs_for_translation()
                if self.stop_requested:
                    return False
            
            # Check for large EPUBs and set optimization parameters
            epub_files = (
                []
                if getattr(self, '_metadata_only_run', False)
                else [
                    f for f in self.selected_files
                    if f.lower().endswith('.epub')
                ]
            )
            
            for epub_path in epub_files:
                try:
                    import zipfile
                    with zipfile.ZipFile(epub_path, 'r') as zf:
                        # Quick count without reading content
                        html_files = [f for f in zf.namelist() if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                        file_count = len(html_files)
                        
                        if file_count > 50:
                            self.append_log(f"📚 Large EPUB detected: {file_count} chapters")
                            
                            # Get user-configured worker count
                            if hasattr(self, 'config') and 'extraction_workers' in self.config:
                                max_workers = self.config.get('extraction_workers', 2)
                            else:
                                # Fallback to environment variable or default
                                max_workers = int(os.environ.get('EXTRACTION_WORKERS', '2'))
                            
                            # Set extraction parameters
                            os.environ['EXTRACTION_WORKERS'] = str(max_workers)
                            os.environ['EXTRACTION_PROGRESS_CALLBACK'] = 'enabled'
                            
                            # Set progress interval based on file count
                            if file_count > 500:
                                progress_interval = 50
                                os.environ['EXTRACTION_BATCH_SIZE'] = '100'
                                self.append_log(f"⚡ Using {max_workers} workers with batch size 100")
                            elif file_count > 200:
                                progress_interval = 25
                                os.environ['EXTRACTION_BATCH_SIZE'] = '50'
                                self.append_log(f"⚡ Using {max_workers} workers with batch size 50")
                            elif file_count > 100:
                                progress_interval = 20
                                os.environ['EXTRACTION_BATCH_SIZE'] = '25'
                                self.append_log(f"⚡ Using {max_workers} workers with batch size 25")
                            else:
                                progress_interval = 10
                                os.environ['EXTRACTION_BATCH_SIZE'] = '20'
                                self.append_log(f"⚡ Using {max_workers} workers with batch size 20")
                            
                            os.environ['EXTRACTION_PROGRESS_INTERVAL'] = str(progress_interval)
                            
                            # Enable performance flags for large files
                            os.environ['FAST_EXTRACTION'] = '1'
                            os.environ['PARALLEL_PARSE'] = '1'
                            
                except Exception as e:
                    # If we can't check, just continue
                    pass
            
            # Set essential environment variables from current config before translation
            os.environ['BATCH_TRANSLATE_HEADERS'] = '1' if self.config.get('batch_translate_headers', True) else '0'
            os.environ['IGNORE_HEADER'] = '1' if self.config.get('ignore_header', False) else '0'
            os.environ['ALLOW_AI_MARKDOWN_HEADERS'] = '1' if self.config.get('allow_ai_markdown_headers', False) else '0'
            skip_title_tag = bool(self.config.get('skip_title_tag_translation', False))
            os.environ['SKIP_TITLE_TAG_TRANSLATION'] = '1' if skip_title_tag else '0'
            os.environ['USE_TITLE'] = '0' if skip_title_tag else '1'
            try:
                import large_env
                large_env.set_env(
                    'IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT',
                    str(self.config.get(
                        'image_only_title_tag_system_prompt',
                        DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT,
                    ) or DEFAULT_IMAGE_ONLY_TITLE_TAG_SYSTEM_PROMPT),
                )
            except Exception:
                pass
            os.environ['REMOVE_DUPLICATE_H1_P'] = '1' if self.config.get('remove_duplicate_h1_p', False) else '0'
            os.environ['FIX_STRAY_P_GT_EPUB'] = '1' if self.config.get('fix_stray_p_gt_epub', False) else '0'
            os.environ['FIX_STRAY_P_GT_BS'] = '1' if self.config.get('fix_stray_p_gt_bs', False) else '0'
            os.environ['USE_SORTED_FALLBACK'] = '1' if self.config.get('use_sorted_fallback', False) else '0'
            # Update temperature and max output tokens from GUI's current values
            os.environ['TRANSLATION_TEMPERATURE'] = str(self.trans_temp.text())
            os.environ['DISABLE_TEMPERATURE'] = '1' if self.disable_temperature_var else '0'
            os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
            # Set batch header translation prompts from config
            _output_lang = self.config.get('output_language', 'English')
            os.environ['BATCH_HEADER_SYSTEM_PROMPT'] = self.config.get('batch_header_system_prompt', '').replace('{target_lang}', _output_lang)
            os.environ['BATCH_HEADER_PROMPT'] = self.config.get('batch_header_prompt', '').replace('{target_lang}', _output_lang)
            os.environ['BATCH_HEADER_PREPEND_NUMBER_PATTERN'] = str(
                self.config.get('batch_header_prepend_number_pattern', '') or '')
            os.environ['OUTPUT_LANGUAGE'] = _output_lang
            
            # ===== PRE-TRANSLATION GLOSSARY EXTRACTION (Balanced/Full modes) =====
            auto_glossary_mode = self._current_auto_glossary_mode()

            if getattr(self, '_metadata_only_run', False) and auto_glossary_mode in ('balanced', 'full'):
                self.append_log("🌐 Metadata-only mode: skipping auto glossary extraction")
                auto_glossary_mode = 'off'
            elif getattr(self, '_single_chapter_filter', None) and auto_glossary_mode in ('balanced', 'full'):
                self.append_log("🎯 Single-chapter mode: skipping auto glossary extraction (jumping straight to translation)")
                auto_glossary_mode = 'off'

            current_output_mode = self._active_translation_output_mode()
            if current_output_mode == 'refinement' and auto_glossary_mode in ('balanced', 'full'):
                self.append_log("✨ Skipping auto glossary extraction for refinement mode")
                auto_glossary_mode = 'off'
            elif current_output_mode == 'audio' and auto_glossary_mode in ('balanced', 'full'):
                self.append_log("📑 Skipping auto glossary extraction for Audio output mode")
                auto_glossary_mode = 'off'
            
            if current_output_mode == 'vision' and auto_glossary_mode in ('balanced', 'full'):
                glossary_merging_enabled, glossary_merge_count, glossary_chapter_split = self._current_glossary_request_env(
                    force_balanced_request_merging=(auto_glossary_mode == 'balanced')
                )
                os.environ['GLOSSARY_REQUEST_MERGING_ENABLED'] = glossary_merging_enabled
                os.environ['GLOSSARY_ENABLE_CHAPTER_SPLIT'] = glossary_chapter_split
                os.environ['GLOSSARY_REQUEST_MERGE_COUNT'] = glossary_merge_count
                self.append_log(f"📑 Vision mode: OCR prepass will run before glossary extraction (merge count: {glossary_merge_count})")
            elif auto_glossary_mode in ('balanced', 'full'):
                # Check if a glossary was MANUALLY loaded by the user for this file
                # Auto-loaded glossaries (from autofill) could be incomplete from a stopped extraction
                has_existing_glossary = bool(
                    getattr(self, 'manual_glossary_path', None) and 
                    os.path.exists(getattr(self, 'manual_glossary_path', '')) and
                    getattr(self, 'manual_glossary_manually_loaded', False)
                )
                
                if not has_existing_glossary:
                    # Clear any auto-loaded glossary before re-extraction
                    # (it may be incomplete from a previous stopped extraction)
                    if (getattr(self, 'manual_glossary_path', None) and 
                        not getattr(self, 'manual_glossary_manually_loaded', False)):
                        self.append_log(f"📑 Clearing auto-loaded glossary for fresh extraction")
                        self.manual_glossary_path = None
                        os.environ.pop('MANUAL_GLOSSARY', None)
                    
                    mode_display = auto_glossary_mode.capitalize()
                    self.append_log(f"\n{'='*60}")
                    self.append_log(f"📑 Auto Glossary Mode: {mode_display}")
                    
                    # Apply invisible hardcoded overrides for Balanced mode
                    # (text/EPUB-specific; skip for image-only inputs)
                    saved_env = {}
                    image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
                    has_text_files = any(
                        os.path.splitext(f)[1].lower() not in image_exts
                        for f in getattr(self, 'selected_files', [])
                    )
                    if auto_glossary_mode == 'balanced' and has_text_files:
                        _balanced_merge_enabled, _balanced_merge_count, _balanced_chapter_split = self._current_glossary_request_env(
                            force_balanced_request_merging=True
                        )
                        split_status = "enabled" if _balanced_chapter_split == '1' else "disabled"
                        self.append_log(f"📑 Balanced mode: request merging enabled (99), dynamic request splitting {split_status}")
                        self.append_log(f"📑 Request merging is hardcoded for optimal glossary quality")
                        saved_env = {
                            'GLOSSARY_REQUEST_MERGING_ENABLED': os.environ.get('GLOSSARY_REQUEST_MERGING_ENABLED'),
                            'GLOSSARY_REQUEST_MERGE_COUNT': os.environ.get('GLOSSARY_REQUEST_MERGE_COUNT'),
                            'GLOSSARY_ENABLE_CHAPTER_SPLIT': os.environ.get('GLOSSARY_ENABLE_CHAPTER_SPLIT'),
                        }
                        os.environ['GLOSSARY_REQUEST_MERGING_ENABLED'] = _balanced_merge_enabled
                        os.environ['GLOSSARY_REQUEST_MERGE_COUNT'] = _balanced_merge_count
                        os.environ['GLOSSARY_ENABLE_CHAPTER_SPLIT'] = _balanced_chapter_split
                    elif auto_glossary_mode == 'balanced':
                        self.append_log(f"📑 Balanced mode: image glossary extraction")
                    elif has_text_files:  # full + text
                        self.append_log(f"📑 Full mode: chapter-by-chapter extraction (most thorough)")
                    else:  # full + images only
                        self.append_log(f"📑 Full mode: image glossary extraction")
                    
                    self.append_log(f"📑 Running glossary extraction before translation...")
                    self.append_log(f"{'='*60}")
                    
                    try:
                        # Reuse the exact same Extract Glossary flow
                        self._glossary_stop_was_requested = False  # Reset before extraction
                        self.run_glossary_extraction_direct(
                            force_balanced_request_merging=(auto_glossary_mode == 'balanced' and has_text_files)
                        )

                        from glossary_translation_gate import (
                            progress_path_for_source,
                            retryable_glossary_qa_failures,
                        )

                        glossary_root = os.path.abspath(
                            os.environ.get('OUTPUT_DIRECTORY')
                            or self.config.get('output_directory')
                            or os.getcwd()
                        )
                        skip_api_errors = self._live_bool_setting(
                            'glossary_skip_api_error_retries_checkbox',
                            'glossary_skip_api_error_retries_var',
                            'glossary_skip_api_error_retries',
                            False,
                        )
                        max_attempts = self._resolve_max_retries()
                        for attempt in range(2, max_attempts + 1):
                            if self.stop_requested or getattr(self, '_glossary_stop_was_requested', False):
                                break
                            failed_count = 0
                            for source in self.selected_files:
                                source_root = (
                                    os.path.dirname(os.path.abspath(source))
                                    if sys.platform == 'darwin'
                                    and not (os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory'))
                                    else glossary_root
                                )
                                progress_path = progress_path_for_source(source, source_root)
                                failed_count += len(retryable_glossary_qa_failures(
                                    progress_path, skip_api_errors=skip_api_errors,
                                ))
                            if not failed_count:
                                break
                            self.append_log(
                                f"🔄 Retrying {failed_count} failed glossary chapter(s) "
                                f"(attempt {attempt}/{max_attempts})..."
                            )
                            self.run_glossary_extraction_direct(
                                force_balanced_request_merging=(auto_glossary_mode == 'balanced' and has_text_files)
                            )
                        
                        # Check saved stop flag (run_glossary_extraction_direct resets self.stop_requested in finally)
                        if self.stop_requested or getattr(self, '_glossary_stop_was_requested', False):
                            self.append_log("⏹️ Translation cancelled during glossary extraction")
                            # Clear auto-loaded glossary so next run re-extracts
                            if not getattr(self, 'manual_glossary_manually_loaded', False):
                                self.manual_glossary_path = None
                                os.environ.pop('MANUAL_GLOSSARY', None)
                            return False
                        
                        # Auto-load the generated glossary (only if extraction completed fully)
                        generated_glossary = self._auto_load_glossary_after_extraction()

                        if getattr(self, '_input_output_run_active', False):
                            self.append_log(
                                "⏸️ Direct Text: glossary generation is complete; "
                                "waiting for approval before translation"
                            )
                            if not self._await_direct_text_glossary_approval(
                                generated_glossary
                                or getattr(self, 'manual_glossary_path', '')
                            ):
                                self.append_log(
                                    "⏹️ Direct Text translation cancelled at "
                                    "the glossary approval step"
                                )
                                return False

                        self.append_log(f"\n📑 Glossary extraction complete, proceeding to translation...")
                    except Exception as e:
                        self.append_log(f"⚠️ Glossary extraction failed: {e}")
                        self.append_log(f"📑 Continuing translation without auto-generated glossary")
                    finally:
                        # Restore env vars for Balanced mode
                        if saved_env:
                            for key, val in saved_env.items():
                                if val is None:
                                    os.environ.pop(key, None)
                                else:
                                    os.environ[key] = val
                else:
                    self.append_log(f"📑 Glossary already loaded, skipping auto-extraction")
            # ===== END PRE-TRANSLATION GLOSSARY EXTRACTION =====

            # Auto-mapping keeps its authoritative glossary in the shared
            # Glossary/<book>/ folder. Mirror that latest resolved file into
            # the translation output now so an older output-side glossary
            # never remains visible after the user starts a run.
            self._sync_automapped_glossaries_to_output()

            if (
                auto_glossary_mode in ('balanced', 'full')
                and current_output_mode != 'vision'
                and self._live_bool_setting(
                    'glossary_require_complete_checkbox',
                    'glossary_require_complete_before_translation_var',
                    'glossary_require_complete_before_translation',
                    False,
                )
            ):
                from glossary_translation_gate import glossary_complete, progress_path_for_source

                glossary_root = os.path.abspath(
                    os.environ.get('OUTPUT_DIRECTORY')
                    or self.config.get('output_directory')
                    or os.getcwd()
                )
                for source in self.selected_files:
                    source_root = (
                        os.path.dirname(os.path.abspath(source))
                        if sys.platform == 'darwin'
                        and not (os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory'))
                        else glossary_root
                    )
                    progress_path = progress_path_for_source(source, source_root)
                    base = os.path.splitext(os.path.basename(source))[0]
                    glossary_path = next((
                        os.path.join(os.path.dirname(progress_path), f'{base}_glossary{ext}')
                        for ext in ('.json', '.csv', '.txt', '.md')
                        if progress_path and os.path.isfile(
                            os.path.join(os.path.dirname(progress_path), f'{base}_glossary{ext}')
                        )
                    ), None)
                    ready, reason = glossary_complete(
                        progress_path,
                        glossary_path,
                        dict(
                            self.config,
                            custom_entry_types=(
                                getattr(self, 'custom_entry_types', None)
                                or self.config.get('custom_entry_types', {})
                            ),
                        ),
                        require_minimal_pass=self._glossary_add_minimal_pass_env_value() == '1',
                        is_epub=str(source).lower().endswith('.epub'),
                    )
                    if not ready:
                        self.append_log(
                            f"⏸️ Translation blocked for {os.path.basename(source)}: "
                            f"glossary is below 100% ({reason})."
                        )
                        return False

            # Call the direct function
            if getattr(self, '_translation_run_is_multipass_qa_refinement', False):
                mode_label = str(getattr(self, '_translation_run_qa_refinement_mode', '') or 'multipass').title()
                self.append_log(f"🚀 Starting {mode_label} multipass refinement...")
            elif getattr(self, '_metadata_only_run', False):
                self.append_log("🌐 Starting metadata translation...")
            else:
                self.append_log("🚀 Starting translation...")
            translation_completed = self.run_translation_direct()
            if (
                translation_completed
                and not self.stop_requested
                and getattr(self, '_translation_run_followup_translation_after_refinement', False)
            ):
                skipped_failures = list(getattr(self, '_translation_run_skipped_qa_failures', []))
                skipped_chapters = [failure["chapter"] for failure in skipped_failures]
                self._clear_translation_run_overrides()
                self.append_log(
                    "🚀 Starting regular translation retry for skipped QA-failed entries "
                    f"(chapters: {self._format_chapter_list(skipped_chapters)})"
                )
                translation_completed = self.run_translation_direct()
            
            # Post-translation scanning phase
            # If scanning phase toggle is enabled, launch scanner after translation
            # BUT only if translation completed successfully (not stopped by user)
            # SKIP scanning for CSV/JSON files (they are plain text glossaries, not translations)
            # SKIP scanning for image files (they are image translations, not text documents)
            try:
                # Check if any of the files are CSV/JSON (but NOT TXT - TXT files are valid translation sources)
                csv_json_files = [f for f in self.selected_files if f.lower().endswith(('.csv', '.json'))]
                
                # Check if any files are images
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
                image_files = [f for f in self.selected_files if os.path.splitext(f)[1].lower() in image_extensions]
                sdlxliff_files = [f for f in self.selected_files if f.lower().endswith('.sdlxliff')]
                subtitle_files = [
                    f for f in self.selected_files
                    if f.lower().endswith(('.srt', '.ass', '.lrc'))
                ]
                
                if csv_json_files:
                    self.append_log("📑 Skipping post-translation scanning for CSV/JSON files")
                elif image_files:
                    self.append_log("🖼️ Skipping post-translation scanning for image files")
                if sdlxliff_files:
                    self.append_log("SDLXLIFF: skipping post-translation scanner")
                if subtitle_files:
                    self.append_log("Subtitles: skipping post-translation scanner")
                current_run_output_mode = self._active_translation_output_mode()
                if getattr(self, '_metadata_only_run', False):
                    self.append_log("🌐 Metadata-only mode: skipping post-translation QA scan")
                elif csv_json_files or image_files or sdlxliff_files or subtitle_files:
                    pass
                elif current_run_output_mode == 'refinement':
                    self.append_log("✨ Skipping post-translation scanning for refinement mode")
                elif current_run_output_mode == 'image':
                    self.append_log("🖼️ Skipping post-translation scanning for image output mode")
                elif (not getattr(self, '_input_output_run_active', False)
                      and hasattr(self, 'scan_phase_enabled_var')
                      and self.scan_phase_enabled_var
                      and translation_completed
                      and not self.stop_requested):
                    mode = self._get_scan_phase_mode()
                    self.append_log(f"🧪 Scanning phase enabled — launching QA Scanner in {mode} mode...")
                    # Emit signal to trigger QA scan on main thread
                    self._ui_request('trigger_qa_scan')
            except Exception as e:
                self.append_log(f"⚠️ Could not launch post-translation scan: {e}")
                import traceback
                self.append_log(traceback.format_exc())
            
            return translation_completed
        except Exception as e:
            self.append_log(f"❌ Error in thread: {e}")
            import traceback
            self.append_log(traceback.format_exc())
            return False
        finally:
            # Clean up environment variables
            env_vars = [
                'EXTRACTION_WORKERS', 'EXTRACTION_BATCH_SIZE',
                'EXTRACTION_PROGRESS_CALLBACK', 'EXTRACTION_PROGRESS_INTERVAL',
                'FAST_EXTRACTION', 'PARALLEL_PARSE'
            ]
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]

            # Single-chapter mode is strictly per-run — never leak the
            # filter (or forced streaming) into the next regular run.
            self._single_chapter_filter = None
            self._force_stream_all = False
            os.environ.pop('SINGLE_CHAPTER_FILTER', None)
            self._metadata_only_run = False
            self._metadata_output_roots = {}
            self._metadata_worker_processes = set()
            os.environ.pop('METADATA_ONLY', None)

            self._clear_translation_run_overrides()
            self._single_qa_resolution_request = None
            # Reset stop flags ONCE, at the true end of the (possibly
            # multi-phase) run. Moved here from run_translation_direct() so a
            # chained phase (refinement → followup translation) does not reset
            # Stop mid-run. The next run's setup also resets these, so this is
            # just clean end-of-run state.
            self.stop_requested = False
            try:
                translation_stop_flag = self._backend_entry('translation_stop_flag')
                if translation_stop_flag:
                    translation_stop_flag(False)
                if (hasattr(self, '_main_module') and self._main_module
                        and hasattr(self._main_module, 'set_stop_flag')):
                    self._main_module.set_stop_flag(False)
            except Exception:
                pass
            self.translation_thread = None
            # Emit signal to update button (thread-safe)
            self._ui_request('thread_complete')

    def run_translation_direct(self):
        """Run translation directly - handles multiple files and different file types"""

        backend_glossary_callback_installed = False
        try:
            if not str(getattr(self, 'model_var', '') or '').strip():
                self.append_log("❌ Translation stopped: no model is selected.")
                return False

            self._export_multipass_runtime_env()

            if getattr(self, '_input_output_run_active', False):
                try:
                    import TransateKRtoEN
                    TransateKRtoEN.set_direct_text_glossary_approval_callback(
                        self._await_direct_text_glossary_approval
                    )
                    backend_glossary_callback_installed = True
                except Exception as exc:
                    self.append_log(
                        f"⚠️ Could not install Direct Text glossary approval gate: {exc}"
                    )

            # AUTO-SWITCH PROFILE BASED ON EXTRACTION MODE
            # Check if profile name contains BeautifulSoup or html2text
            current_profile = self.profile_var
            if current_profile:
                profile_lower = current_profile.lower()
                
                # Check if profile indicates an extraction mode
                if 'beautifulsoup' in profile_lower:
                    # Switch to BeautifulSoup extraction mode
                    if hasattr(self, 'text_extraction_method_var'):
                        self.text_extraction_method_var = 'standard'
                        self.append_log(f"🔄 Auto-switched to BeautifulSoup extraction (profile: {current_profile})")
                elif 'html2text' in profile_lower:
                    # Switch to html2text extraction mode  
                    if hasattr(self, 'text_extraction_method_var'):
                        self.text_extraction_method_var = 'enhanced'
                        self.append_log(f"🔄 Auto-switched to html2text extraction (profile: {current_profile})")
            
            # Re-attach GUI logging handlers to reclaim logs from standalone header translation
            try:
                self._attach_gui_logging_handlers()
            except Exception:
                pass
            
            # Restore print hijack if it was captured by manga translator
            # This ensures main GUI logs go to main GUI, not manga GUI
            try:
                import builtins
                # Check if print was hijacked by manga translator
                if hasattr(builtins, '_manga_log_callbacks') and builtins._manga_log_callbacks:
                    # Restore original print for main GUI
                    if hasattr(builtins, 'print') and hasattr(builtins.print, '__name__'):
                        if builtins.print.__name__ == 'manga_print':
                            # Print is hijacked, restore it
                            from manga_translator import MangaTranslator
                            if hasattr(MangaTranslator, '_original_print_backup'):
                                builtins.print = MangaTranslator._original_print_backup
                                # Also restore in unified_api_client
                                try:
                                    import sys
                                    import unified_api_client
                                    uc_module = sys.modules.get('unified_api_client')
                                    if uc_module:
                                        uc_module.__dict__['print'] = MangaTranslator._original_print_backup
                                except Exception:
                                    pass
            except Exception:
                pass
            
            # Check stop at the very beginning
            if self.stop_requested:
                return False
            
            # DON'T CALL _lazy_load_modules HERE!
            # Modules are already loaded in the wrapper
            # Just verify they're loaded
            if not self._modules_loaded:
                self.append_log("❌ Translation modules not loaded")
                return False

            # Check stop after verification
            if self.stop_requested:
                return False
            # Sync streaming toggle to env for all backends.
            # ``_force_stream_all`` (set by the EPUB Reader's live "Translate"
            # action) overrides every streaming toggle to ON for this run,
            # regardless of the user's persisted settings.
            _force_stream = bool(getattr(self, '_force_stream_all', False))
            try:
                stream_on = _force_stream or bool(getattr(self, 'enable_streaming_var', self.config.get('enable_streaming', False)))
                os.environ['ENABLE_STREAMING'] = '1' if stream_on else '0'
                self.append_log(f"🛰️ Streaming {'enabled' if stream_on else 'disabled'} (exported ENABLE_STREAMING)")
            except Exception:
                pass
            try:
                allow_batch_logs = _force_stream or bool(getattr(self, 'allow_batch_stream_logs_var', self.config.get('allow_batch_stream_logs', False)))
                os.environ['ALLOW_BATCH_STREAM_LOGS'] = '1' if allow_batch_logs else '0'
            except Exception:
                pass
            try:
                allow_authgpt_logs = _force_stream or bool(getattr(self, 'allow_authgpt_batch_stream_logs_var', self.config.get('allow_authgpt_batch_stream_logs', False)))
                os.environ['ALLOW_AUTHGPT_BATCH_STREAM_LOGS'] = '1' if allow_authgpt_logs else '0'
            except Exception:
                pass
            try:
                stream_thinking = _force_stream or bool(getattr(self, 'stream_thinking_logs_var', self.config.get('stream_thinking_logs', False)))
                os.environ['STREAM_THINKING_LOGS'] = '1' if stream_thinking else '0'
            except Exception:
                pass
            if _force_stream:
                self._apply_forced_streaming_environment()
                self.append_log("🛰️ Live view: all streaming toggles forced ON for this run")

            # SET GLOSSARY IN ENVIRONMENT
            # If a per-input mapping exists, we set MANUAL_GLOSSARY per file inside the loop.
            use_glossary_map = False
            try:
                mgm = getattr(self, 'manual_glossary_map', None)
                use_glossary_map = isinstance(mgm, dict) and any(v for v in mgm.values())
            except Exception:
                use_glossary_map = False

            if use_glossary_map:
                os.environ.pop('MANUAL_GLOSSARY', None)
                try:
                    mapped = len([1 for _k, _v in (self.manual_glossary_map or {}).items() if _v])
                except Exception:
                    mapped = 0
                self.append_log(f"📑 Glossary mapping enabled ({mapped} file(s) mapped)")
            else:
                if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                    os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                    self.append_log(f"📑 Set glossary in environment: {os.path.basename(self.manual_glossary_path)}")
                else:
                    # Clear any previous glossary from environment
                    os.environ.pop('MANUAL_GLOSSARY', None)
                    self.append_log(f"ℹ️ No glossary loaded")

            # ========== NEW: APPLY OPF-BASED SORTING ==========
            if (
                self._has_epub_conversion_inputs()
                and not getattr(self, '_zip_inputs_resolved_for_current_run', False)
            ):
                self.append_log("📦 Preparing archive/HTML input(s) before file processing...")
                self._resolve_zip_inputs_for_translation()
                if self.stop_requested:
                    return False

            # Sort files based on OPF order if available
            original_file_count = len(self.selected_files)
            has_epub_sources = any(
                os.path.splitext(os.fspath(path))[1].casefold() == '.epub'
                for path in self.selected_files
            )
            if getattr(self, '_metadata_only_run', False):
                self.append_log(
                    f"🌐 Processing {original_file_count} EPUB metadata "
                    "source(s) in selection order"
                )
            else:
                self.selected_files = self._get_opf_file_order(
                    self.selected_files
                )
                if has_epub_sources:
                    self.append_log(
                        f"📚 Processing {original_file_count} source file(s) in "
                        "selection order; each EPUB uses its own internal OPF spine"
                    )
                else:
                    self.append_log(
                        f"📚 Processing {original_file_count} files in reading order"
                    )
            # ====================================================

            # ── Generative-only mode (no input file) ───────────────────────
            # When the model indicates image or video generation and no real file
            # was selected, skip the normal file loop and fire a single API call
            # using the system-prompt / translation-chunk-prompt as the user prompt.
            _active_model = str(getattr(self, 'model_var', ''))
            _is_gen_mode = (
                len(self.selected_files) == 1
                and self.selected_files[0] == "__generative_mode__"
            ) or (
                (self._model_is_image_gen(_active_model) or self._model_is_video_gen(_active_model) or self._is_generative_output_mode())
                and not any(
                    os.path.exists(p) for p in (self.selected_files or [])
                    if p != "__generative_mode__"
                )
            )

            if _is_gen_mode:
                return self._run_generative_prompt_mode()

            # Process each file
            total_files = len(self.selected_files)
            successful = 0
            failed = 0

            # Honor OUTPUT_DIRECTORY override globally for this run
            try:
                override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
                if override_dir:
                    os.environ['OUTPUT_DIRECTORY'] = os.path.abspath(override_dir)
                    os.environ['OUTPUT_DIR'] = os.path.abspath(override_dir)
                    self.append_log(f"📁 Using output override: {os.environ['OUTPUT_DIRECTORY']}")
            except Exception as e:
                self.append_log(f"⚠️ Could not apply OUTPUT_DIRECTORY override: {e}")

            if (
                getattr(self, '_metadata_only_run', False)
                and total_files > 1
                and bool(getattr(
                    self,
                    'batch_translation_var',
                    self.config.get('batch_translation', True),
                ))
            ):
                return self._run_parallel_metadata_files(self.selected_files)
            
            # Check if we're processing multiple images - if so, create a combined output folder
            image_extensions = set(IMAGE_ATTACHMENT_EXTENSIONS)
            video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
            media_extensions = image_extensions | video_extensions
            image_files = [f for f in self.selected_files if os.path.splitext(f)[1].lower() in media_extensions]
            
            combined_image_output_dir = None
            if len(image_files) > 1:
                # Check stop before creating directories
                if self.stop_requested:
                    return False
                    
                # Get the common parent directory name or use timestamp
                parent_dir = os.path.dirname(self.selected_files[0])
                folder_name = os.path.basename(parent_dir) if parent_dir else f"OCR_{int(time.time())}"
                
                # Check for output directory override
                override_dir = os.environ.get('OUTPUT_DIRECTORY') or self.config.get('output_directory')
                if override_dir:
                    combined_image_output_dir = os.path.join(override_dir, folder_name)
                else:
                    combined_image_output_dir = folder_name
                
                os.makedirs(combined_image_output_dir, exist_ok=True)
                
                # Create images subdirectory for originals
                images_dir = os.path.join(combined_image_output_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                self.append_log(f"📁 Created combined output directory: {combined_image_output_dir}")
            
            processed_subtitle_bundle_ids = set()
            for i, file_path in enumerate(self.selected_files):
                if self.stop_requested:
                    # Suppress per-file stop spam; summary will be shown later
                    break

                subtitle_output_info = self._subtitle_zip_output_info(file_path)
                subtitle_bundle_id = (
                    str(subtitle_output_info.get('bundle_id') or '')
                    if subtitle_output_info
                    else ''
                )
                subtitle_bundle_files = (
                    list(subtitle_output_info.get('bundle_files') or [])
                    if subtitle_output_info
                    else []
                )
                if (
                    subtitle_bundle_id
                    and subtitle_bundle_id in processed_subtitle_bundle_ids
                ):
                    continue

                # Apply per-file glossary mapping (if present)
                try:
                    mgm = getattr(self, 'manual_glossary_map', None)
                    if isinstance(mgm, dict) and mgm:
                        key = os.path.normpath(os.path.abspath(file_path))
                        gp = mgm.get(file_path) or mgm.get(key) or mgm.get(os.path.normpath(file_path))
                        if gp:
                            os.environ['MANUAL_GLOSSARY'] = gp
                            # Inform user which glossary is used for this EPUB
                            try:
                                if str(file_path).lower().endswith('.epub'):
                                    self.append_log(
                                        f"📑 Using mapped glossary for {os.path.basename(file_path)}: {os.path.basename(gp)}"
                                    )
                            except Exception:
                                pass
                        else:
                            os.environ.pop('MANUAL_GLOSSARY', None)
                except Exception:
                    pass
                
                self.current_file_index = i
                
                # Log progress for multiple files
                if total_files > 1:
                    self.append_log(f"\n{'='*60}")
                    self.append_log(f"📄 Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
                    progress_percent = ((i + 1) / total_files) * 100
                    self.append_log(f"📊 Overall progress: {progress_percent:.1f}%")
                    self.append_log(f"{'='*60}")

                if (
                    str(file_path).lower().endswith(('.zip', '.cbz', '.html', '.htm', '.xhtml'))
                    and not getattr(self, '_zip_inputs_resolved_for_current_run', False)
                ):
                    old_file_path = file_path
                    file_path = self._convert_zip_input_to_epub_if_needed(file_path)
                    self.selected_files[i] = file_path
                    if file_path != old_file_path:
                        try:
                            self._ui_request('input_files_updated', list(self.selected_files))
                        except Exception:
                            pass
                
                if not os.path.exists(file_path):
                    self.append_log(f"❌ File not found: {file_path}")
                    failed += 1
                    continue
                
                # Determine file type and process accordingly
                ext = os.path.splitext(file_path)[1].lower()
                
                try:
                    if ext in media_extensions:
                        # Process as image/video with combined output directory if applicable
                        if self._process_image_file(file_path, combined_image_output_dir):
                            successful += 1
                        else:
                            failed += 1
                    elif ext == '.exe' or file_path in (getattr(self, RPGMAKER_GAME_INPUTS_ATTR, None) or ()):
                        # Process as RPG Maker game via GTool
                        if self._process_rpgmaker_game(file_path):
                            successful += 1
                        else:
                            failed += 1
                    elif ext in {'.epub', '.txt', '.csv', '.json', '.pdf', '.md', '.sdlxliff', '.srt', '.ass', '.lrc'}:
                        # Process as EPUB/text/PDF/SDLXLIFF/subtitle input.
                        if len(subtitle_bundle_files) > 1:
                            self.append_log(
                                f"📦 Translating {len(subtitle_bundle_files)} subtitle "
                                "file(s) from this ZIP as one parallel batch job"
                            )
                        result = self._process_text_file(file_path)
                        if subtitle_bundle_id and len(subtitle_bundle_files) > 1:
                            processed_subtitle_bundle_ids.add(subtitle_bundle_id)
                            if result:
                                successful += len(subtitle_bundle_files)
                            else:
                                failed += len(subtitle_bundle_files)
                        elif result:
                            successful += 1
                        else:
                            failed += 1
                    else:
                        self.append_log(f"⚠️ Unsupported file type: {ext}")
                        failed += 1
                        
                except Exception as e:
                    self.append_log(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
                    import traceback
                    self.append_log(f"❌ Full error: {traceback.format_exc()}")
                    failed += 1
            
            # Check stop before final summary
            if self.stop_requested:
                self.append_log(f"\n⏹️ Translation stopped - processed {successful} of {total_files} files")
                # Reset progress bar when stopped
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar:
                        self.progress_bar.setValue(0)
                        self.progress_bar.setFormat("Batch stopped")
                        
                    # Also clean up any lingering watchdog files to ensure "in-flight" count clears
                    # ONLY if WAIT_FOR_CHUNKS=0 (immediate stop behavior)
                    if os.environ.get('WAIT_FOR_CHUNKS', '0') == '0':
                        self._reset_api_watchdog_progress(clear_stale_external_files=True)
                        
                        # And ensure hard cancel is triggered one last time
                        import unified_api_client
                        if hasattr(unified_api_client, 'hard_cancel_all'):
                            unified_api_client.hard_cancel_all()
                except Exception:
                    pass
                return False
                
            # Final summary
            try:
                self._log_translation_qa_failure_summary(phase="end")
            except Exception as e:
                self.append_log(f"⚠️ Could not read final QA failure summary: {e}")

            if total_files > 1:
                self.append_log(f"\n{'='*60}")
                self.append_log(f"📊 Translation Summary:")
                self.append_log(f"   ✅ Successful: {successful} files")
                if failed > 0:
                    self.append_log(f"   ❌ Failed: {failed} files")
                self.append_log(f"   📁 Total: {total_files} files")
                
                # Create CBZ if we have generated images
                if hasattr(self, 'generated_images') and self.generated_images:
                    self.append_log(f"\n📦 Creating CBZ archive with {len(self.generated_images)} generated images...")
                    try:
                        import zipfile
                        # Get output directory from first image path
                        output_dir = os.path.dirname(self.generated_images[0])
                        cbz_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}.cbz")
                        
                        # Create CBZ (which is just a ZIP file)
                        with zipfile.ZipFile(cbz_path, 'w', zipfile.ZIP_DEFLATED) as cbz:
                            for img_path in sorted(self.generated_images):
                                # Add image with just the filename (no path)
                                cbz.write(img_path, os.path.basename(img_path))
                        
                        self.append_log(f"✅ CBZ created: {cbz_path}")
                        self.append_log(f"   📁 Contains {len(self.generated_images)} images")
                        
                        # Clear the list for next batch
                        self.generated_images = []
                    except Exception as e:
                        self.append_log(f"❌ Failed to create CBZ: {e}")
                        import traceback
                        self.append_log(traceback.format_exc())
                
                if combined_image_output_dir and successful > 0:
                    self.append_log(f"\n💡 Tip: You can now compile the HTML files in '{combined_image_output_dir}' into an EPUB")
                    
                    # Check for cover image
                    cover_found = False
                    for img_name in ['cover.png', 'cover.jpg', 'cover.jpeg', 'cover.webp']:
                        if os.path.exists(os.path.join(combined_image_output_dir, "images", img_name)):
                            self.append_log(f"   📖 Found cover image: {img_name}")
                            cover_found = True
                            break
                    
                    if not cover_found:
                        # Use first image as cover
                        images_in_dir = os.listdir(os.path.join(combined_image_output_dir, "images"))
                        if images_in_dir:
                            self.append_log(f"   📖 First image will be used as cover: {images_in_dir[0]}")
                
                self.append_log(f"{'='*60}")
            
            # Only return True if at least one file succeeded
            # This prevents QA scanner from running when all files failed
            if successful == 0:
                return False
            
            return True  # Translation completed successfully
            
        except Exception as e:
            self.append_log(f"❌ Translation setup error: {e}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False
        
        finally:
            if backend_glossary_callback_installed:
                try:
                    import TransateKRtoEN
                    TransateKRtoEN.set_direct_text_glossary_approval_callback(None)
                except Exception:
                    pass
            # IMPORTANT: do NOT reset stop flags, null the translation thread, or
            # emit thread_complete_signal here.
            #
            # run_translation_direct() is only ever called from
            # simple_thread_target(), which owns the thread/button lifecycle and
            # performs this cleanup exactly once in its own finally. When a single
            # run chains multiple phases — e.g. Partial.B2 multipass refinement
            # followed by the "regular translation retry" of skipped QA-failed
            # chapters — this finally runs after the FIRST phase and:
            #   • fires thread_complete_signal → update_run_button() flips the run
            #     button to green ("Run Translation") mid-run, during the 2nd
            #     phase's chapter extraction, and
            #   • sets self.stop_requested = False, which wipes a Stop the user
            #     pressed during phase 1 AND makes the followup-gate
            #     (`not self.stop_requested`) always true, so the 2nd phase always
            #     runs and cannot be stopped.
            # Both were the "button turns green / stop doesn't trigger during
            # multipass" reports. Leave lifecycle + stop-flag reset to
            # simple_thread_target()'s finally.
            self.current_file_index = 0

    def _await_direct_text_glossary_approval(self, glossary_path):
        """Block the translation worker until the GUI records Yes/Edit/No."""
        import threading

        request = {
            "event": threading.Event(),
            "accepted": False,
        }
        self._ui_request(
            'direct_text_glossary_approval',
            os.path.abspath(str(glossary_path or "")) if glossary_path else "",
            request,
        )
        while not request["event"].wait(0.1):
            if self.stop_requested:
                return False
        return bool(request.get("accepted", False))
