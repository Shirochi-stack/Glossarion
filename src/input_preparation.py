"""input_preparation: resolve ZIP / CBZ / HTML / subtitle-ZIP inputs before a run (InputPreparationMixin).

Shared GUI-free core (Glossarion mobile rewrite, milestone U3). Moved verbatim out of
``TranslatorGUI`` (``translator_gui.py`` @ 1719fb59), which inherits the mixin:

* ``resolve_input_to_epub(path, *, conversion_dir, should_stop, log, set_active)``: the
  body of ``_convert_zip_input_to_epub_if_needed`` (38150-38328) as a function (owner
  reads became parameters: the conversion folder, the stop check, the log, and the
  ``_zip_conversion_active`` flag); the mixin method passes the owner's values;
* ``_extract_subtitle_zip_input_if_needed`` (38043-38138), ``_has_epub_conversion_inputs``
  (38140-38148) and ``_resolve_zip_inputs_for_translation`` (38330-38426), which
  rewrites ``selected_files`` / ``manual_glossary_map``; its
  ``input_files_updated_signal.emit`` is the ``_ui_request('input_files_updated', files)``
  hook.

Rules: Python 3.10 compatible; never import PySide6, translator_gui or dpi_setup.
"""

import os
import re
import tempfile

from job_runner import JobHooksMixin

__all__ = ["InputPreparationMixin", "resolve_input_to_epub"]


def resolve_input_to_epub(path, *, conversion_dir='', should_stop, log, set_active=None):
    """Resolve archives and standalone HTML documents to EPUB in the worker.

    ZIP/CBZ (EPUB-in-ZIP, HTML chapter archives, image archives incl. nested ones) and
    standalone .html/.htm/.xhtml become an EPUB next to the input or in
    *conversion_dir*; existing up-to-date conversions are reused. Returns the EPUB path,
    or *path* unchanged when it is not an archive/HTML input or could not be converted.
    ``should_stop()`` cancels between steps, ``log(message)`` receives the user-facing
    lines and ``set_active(bool)`` mirrors the desktop's ``_zip_conversion_active`` flag.
    """
    if not path or not str(path).lower().endswith(('.zip', '.cbz', '.html', '.htm', '.xhtml')):
        return path

    is_html_file = str(path).lower().endswith(('.html', '.htm', '.xhtml'))
    # Keep book.html and book.xhtml distinct and never overwrite book.epub.
    output_name = (
        os.path.basename(path) + '.epub'
        if is_html_file else os.path.splitext(os.path.basename(path))[0] + '.epub'
    )
    conversion_dir = str(
        conversion_dir or ''
    ).strip()
    if conversion_dir:
        os.makedirs(conversion_dir, exist_ok=True)
        epub_path = os.path.join(
            conversion_dir,
            output_name,
        )
    else:
        epub_path = os.path.join(os.path.dirname(path), output_name)
    try:
        import shutil
        from image_archive_epub import (
            convert_image_archive_to_epub,
            ImageArchiveConversionCancelled,
            expected_image_archive_chapter_count,
            generated_image_epub_chapter_count,
            is_epub_zip,
            needs_image_archive_group_rebuild,
            scan_image_archive,
        )

        if set_active is not None:
            set_active(True)
        if is_html_file:
            from html_archive_epub import (
                convert_html_file_to_epub,
                generated_html_epub_chapter_count,
            )

            # Preserve unrelated EPUBs even at the standalone wrapper name.
            candidate_base = os.path.splitext(epub_path)[0]
            suffix = 1
            while os.path.exists(epub_path) and not generated_html_epub_chapter_count(epub_path):
                epub_path = f'{candidate_base}.{suffix}.epub'
                suffix += 1
            log(f"📄 Preparing HTML document: {os.path.basename(path)}")
            # Referenced images/styles can change independently of the HTML.
            result = convert_html_file_to_epub(path, epub_path, should_stop=should_stop)
            log(
                f"📄 Prepared HTML document {os.path.basename(path)} → "
                f"{os.path.basename(result.epub_path)}"
            )
            return result.epub_path

        if is_epub_zip(path):
            if should_stop():
                raise ImageArchiveConversionCancelled()
            needs_copy = (
                not os.path.exists(epub_path)
                or not is_epub_zip(epub_path)
                or os.path.getmtime(epub_path) < os.path.getmtime(path)
            )
            if needs_copy:
                log(f"📦 Preparing EPUB ZIP in background: {os.path.basename(path)}")
                if should_stop():
                    raise ImageArchiveConversionCancelled()
                shutil.copy2(path, epub_path)
                log(
                    f"📦 Converted EPUB ZIP {os.path.basename(path)} → {os.path.basename(epub_path)}"
                )
            else:
                log(f"✅ Using existing {os.path.basename(epub_path)}")
            return epub_path

        log(f"📦 Inspecting ZIP input in background: {os.path.basename(path)}")
        from html_archive_epub import (
            convert_html_archive_to_epub,
            generated_html_epub_chapter_count,
            is_html_archive,
        )

        if is_html_archive(path, should_stop=should_stop):
            needs_rebuild = (
                not generated_html_epub_chapter_count(epub_path)
                or os.path.getmtime(epub_path) < os.path.getmtime(path)
            )
            if needs_rebuild:
                log(f"📦 Converting HTML chapter ZIP in background: {os.path.basename(path)}")
                result = convert_html_archive_to_epub(
                    path,
                    epub_path,
                    should_stop=should_stop,
                )
                log(
                    f"📦 Converted HTML chapter ZIP {os.path.basename(path)} → "
                    f"{os.path.basename(result.epub_path)} ({result.chapter_count} chapter(s))"
                )
            else:
                log(f"✅ Using existing HTML chapter EPUB {os.path.basename(epub_path)}")
            return epub_path

        scan = scan_image_archive(path, should_stop=should_stop)
        nested_image_bundle = bool(scan.image_count and scan.nested_archive_count)
        if scan.is_image_archive or nested_image_bundle:
            chapter_count_mismatch = False
            if os.path.exists(epub_path) and is_epub_zip(epub_path):
                try:
                    expected_chapters = expected_image_archive_chapter_count(
                        path,
                        should_stop=should_stop,
                    )
                    actual_chapters = generated_image_epub_chapter_count(epub_path)
                    chapter_count_mismatch = bool(
                        expected_chapters
                        and actual_chapters
                        and expected_chapters != actual_chapters
                    )
                    if chapter_count_mismatch:
                        log(
                            f"📦 Rebuilding image EPUB chapter layout: "
                            f"{actual_chapters} → {expected_chapters} chapter file(s)"
                        )
                except Exception:
                    chapter_count_mismatch = False
            needs_rebuild = (
                not os.path.exists(epub_path)
                or not is_epub_zip(epub_path)
                or needs_image_archive_group_rebuild(epub_path)
                or chapter_count_mismatch
                or os.path.getmtime(epub_path) < os.path.getmtime(path)
            )
            if needs_rebuild:
                log(f"📦 Converting image ZIP in background: {os.path.basename(path)}")
                result = convert_image_archive_to_epub(
                    path,
                    epub_path,
                    allow_unsupported=nested_image_bundle,
                    should_stop=should_stop,
                )
                nested_note = (
                    f", {result.nested_archive_count} nested archive(s)"
                    if result.nested_archive_count else ""
                )
                ignored_note = (
                    f", ignored {result.ignored_entry_count} non-image sidecar(s)"
                    if result.ignored_entry_count else ""
                )
                log(
                    f"📦 Converted image ZIP {os.path.basename(path)} → "
                    f"{os.path.basename(result.epub_path)} "
                    f"({result.image_count} image(s), "
                    f"{result.chapter_count} grouped chapter(s){nested_note}{ignored_note})"
                )
            else:
                log(f"✅ Using existing image EPUB {os.path.basename(epub_path)}")
            return epub_path

        log(
            f"⚠️ ZIP is not an EPUB, HTML chapter archive, or image-only archive: {os.path.basename(path)}"
        )
        if scan.unsupported_entries:
            log(
                "   Unsupported archive entries: "
                + ", ".join(scan.unsupported_entries[:5])
            )
    except Exception as e:
        cancel_cls = locals().get('ImageArchiveConversionCancelled')
        if cancel_cls is not None and isinstance(e, cancel_cls):
            input_label = 'HTML' if is_html_file else 'ZIP'
            log(f"⏹️ {input_label} conversion cancelled: {os.path.basename(path)}")
        else:
            log(f"⚠️ Could not convert {os.path.basename(path)} to .epub: {e}")
    finally:
        if set_active is not None:
            set_active(False)

    return path


class InputPreparationMixin(JobHooksMixin):
    """Input resolution shared by TranslatorGUI and HeadlessOwner (moved verbatim)."""

    def _extract_subtitle_zip_input_if_needed(self, path):
        """Return extracted SRT/ASS/LRC paths or classify a non-subtitle ZIP."""
        if not path or not str(path).lower().endswith('.zip'):
            return None

        try:
            from subtitle_processor import (
                SubtitleArchiveError,
                extract_subtitle_archive,
                plan_subtitle_archive_outputs,
            )
        except Exception as exc:
            self.append_log(f"❌ Subtitle ZIP support is unavailable: {exc}")
            return []

        try:
            from image_archive_epub import is_epub_zip

            # EPUB files are ZIP containers too. Keep EPUB resolution ahead of
            # any incidental subtitle resources stored inside the book.
            if is_epub_zip(path):
                return None

            temp_root = getattr(self, 'subtitle_zip_temp_root', None)
            if not temp_root:
                temp_root = tempfile.mkdtemp(prefix='glossarion_subtitle_zip_')
                self.subtitle_zip_temp_root = temp_root

            archive_stem = os.path.splitext(os.path.basename(path))[0]
            safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', archive_stem).strip('._')
            extraction_dir = tempfile.mkdtemp(
                prefix=f"{safe_stem or 'subtitles'}_",
                dir=temp_root,
            )
            result = extract_subtitle_archive(path, extraction_dir)
            subtitle_files = list(result.get('files') or [])
            if not subtitle_files:
                try:
                    os.rmdir(extraction_dir)
                except OSError:
                    pass
                return None

            override_dir = (
                os.environ.get('OUTPUT_DIRECTORY')
                or self.config.get('output_directory')
            )
            output_base_dir = (
                os.path.abspath(override_dir)
                if override_dir
                else self._get_output_base_dir(path)
            )
            output_plan = plan_subtitle_archive_outputs(
                path,
                subtitle_files,
                output_base_dir,
                work_base_dir=os.path.join(
                    extraction_dir,
                    ".glossarion_subtitle_work",
                ),
            )
            if not hasattr(self, '_subtitle_zip_output_groups'):
                self._subtitle_zip_output_groups = {}
            bundle_id = os.path.normcase(os.path.abspath(path))
            bundle_files = [os.path.abspath(item) for item in subtitle_files]
            bundle_work_dir = os.path.join(
                extraction_dir,
                ".glossarion_subtitle_bundle",
            )
            for member_info in output_plan.values():
                member_info["bundle_id"] = bundle_id
                member_info["bundle_files"] = bundle_files
                member_info["bundle_work_dir"] = bundle_work_dir
            self._subtitle_zip_output_groups.update(output_plan)
            output_group = next(iter(output_plan.values()), {})

            ignored = int(result.get('ignored_count') or 0)
            ignored_note = (
                f"; ignored {ignored} non-subtitle member(s)" if ignored else ""
            )
            self.append_log(
                f"📦 Extracted {len(subtitle_files)} subtitle file(s) from "
                f"{os.path.basename(path)} into one output folder "
                f"'{output_group.get('group_name', 'Subtitles')}'{ignored_note}"
            )
            return subtitle_files
        except SubtitleArchiveError as exc:
            self.append_log(
                f"❌ Subtitle ZIP rejected: {os.path.basename(path)} — {exc}"
            )
            return []
        except Exception as exc:
            self.append_log(
                f"❌ Could not inspect subtitle ZIP {os.path.basename(path)}: {exc}"
            )
            return []

    def _has_epub_conversion_inputs(self):
        """Include original HTML sources when a previous run prepared EPUBs."""
        html_sources = getattr(self, '_html_epub_source_paths', {}) or {}
        return any(
            str(html_sources.get(path, path)).lower().endswith(
                ('.zip', '.cbz', '.html', '.htm', '.xhtml')
            )
            for path in getattr(self, 'selected_files', []) or []
        )

    def _convert_zip_input_to_epub_if_needed(self, path):
        """Resolve archives and standalone HTML documents to EPUB in the worker."""
        return resolve_input_to_epub(
            path,
            conversion_dir=getattr(self, '_direct_text_archive_conversion_dir', ''),
            should_stop=lambda: bool(getattr(self, 'stop_requested', False)),
            log=self.append_log,
            set_active=lambda active: setattr(self, '_zip_conversion_active', active),
        )

    def _resolve_zip_inputs_for_translation(self):
        """Expand subtitle ZIPs and prepare archives or standalone HTML for processing."""
        files = list(getattr(self, 'selected_files', []) or [])
        if not files:
            self._zip_inputs_resolved_for_current_run = True
            return files

        resolved = []
        persisted_inputs = []
        replacement_pairs = []
        changed = False
        for selected_path in files:
            path = (getattr(self, '_html_epub_source_paths', {}) or {}).get(
                selected_path, selected_path
            )
            if getattr(self, 'stop_requested', False):
                resolved.append(path)
                persisted_inputs.append(path)
                replacement_pairs.append((path, path))
                continue

            subtitle_paths = self._extract_subtitle_zip_input_if_needed(path)
            if subtitle_paths is not None:
                if subtitle_paths:
                    resolved.extend(subtitle_paths)
                    replacement_pairs.extend(
                        (path, subtitle_path) for subtitle_path in subtitle_paths
                    )
                    changed = True
                else:
                    # Keep a rejected archive visible so the run reports the
                    # invalid input instead of silently dropping the selection.
                    resolved.append(path)
                    replacement_pairs.append((path, path))
                persisted_inputs.append(path)
                continue

            new_path = self._convert_zip_input_to_epub_if_needed(path)
            is_html_file = str(path).lower().endswith(('.html', '.htm', '.xhtml'))
            if is_html_file and new_path != path:
                html_sources = dict(getattr(self, '_html_epub_source_paths', {}) or {})
                html_sources[new_path] = path
                self._html_epub_source_paths = html_sources
            resolved.append(new_path)
            persisted_inputs.append(path if is_html_file else new_path)
            replacement_pairs.append((path, new_path))
            changed = changed or (new_path != selected_path)

        if changed:
            self.selected_files = resolved
            self.file_path = resolved[0] if resolved else None

            try:
                manual_map = getattr(self, 'manual_glossary_map', None)
                if isinstance(manual_map, dict) and manual_map:
                    updated_map = dict(manual_map)
                    for old_path, new_path in replacement_pairs:
                        if old_path == new_path:
                            continue
                        old_keys = {
                            old_path,
                            os.path.normpath(old_path),
                            os.path.normpath(os.path.abspath(old_path)),
                        }
                        glossary_path = next(
                            (updated_map[key] for key in old_keys if key in updated_map and updated_map[key]),
                            None,
                        )
                        if glossary_path:
                            updated_map[new_path] = glossary_path
                            updated_map[os.path.normpath(os.path.abspath(new_path))] = glossary_path
                    self.manual_glossary_map = updated_map
                    self.config['manual_glossary_map'] = updated_map
            except Exception:
                pass

            try:
                # Temporary subtitle members are valid only for this app
                # session. Persist their source ZIP so startup can re-extract.
                self.config['last_input_files'] = persisted_inputs
                epub_files = [p for p in resolved if str(p).lower().endswith('.epub')]
                if epub_files:
                    self.selected_epub_path = epub_files[0]
                    self.selected_epub_files = epub_files
                    self.config['last_epub_path'] = epub_files[0]
                    os.environ['EPUB_PATH'] = epub_files[0]
                self.save_config(show_message=False)
            except Exception:
                pass

            try:
                self._ui_request('input_files_updated', resolved)
            except Exception:
                pass

        self._zip_inputs_resolved_for_current_run = True
        return resolved
