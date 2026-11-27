"""Interactive Images: the step itself (GUI-free; the desktop tab and the mobile job kind both call it).

``run_interactive_images(config, output_dir, keys=None, ...)`` works chapter by chapter on the
translated output of one book. It runs after translation as its own step: a chapter that fails or
is rejected leaves its file as it was; a chapter file is only rewritten when its visible text is
exactly the same as before. Rerunnable per chapter: a rerun first strips Glossarion's earlier
insertions, and images keep fixed names (``Glossarion_ch012_1.png``) so they are replaced.

Order per chapter (cheapest checks first): plan (text model) -> place the plan with placeholder
image sources -> text check -> generate the images (image model, with reference images) -> drop the
figures of failed images -> text check again -> atomic write -> progress entry.

The requests use ``unified_api_client.UnifiedClient`` like the custom image edit
(``local_inpainter``): the image request is forced into image output on its own client and uses the
Image Gen / Edit key pool (``context='image_generation'``) when that pool is on. Nothing is added to
any request beyond what the active profile's prompts and their placeholders expand to.
"""

import base64
import os
import re
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import interactive_css_presets as presets
import interactive_images_core as core
import interactive_images_progress_core as ipc

PLAN_CONTEXT = 'interactive_images_plan'
IMAGE_CONTEXT = 'image_generation'
CSS_CONTEXT = 'interactive_images_css'

_PENDING_SRC = 'glx-pending-{k}'


@dataclass
class RunResult:
    processed: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    skipped: int = 0
    images: int = 0
    stopped: bool = False
    changed_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class _Stop(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------------------------

def _default_client_factory(kind, model, config, output_dir):
    from unified_api_client import UnifiedClient

    client = UnifiedClient(model=model, api_key=str(config.get('api_key', '') or ''), output_dir=output_dir)
    if kind == 'image':
        client._force_image_output_mode = True
        settings = core.settings_from_config(config)
        client._forced_image_output_resolution = settings.resolution
        if settings.aspect_ratio and settings.aspect_ratio != 'auto':
            client._forced_image_output_aspect_ratio = settings.aspect_ratio
        client._suppress_custom_image_edit_endpoint = True
    return client


def configure_image_key_pool(config, log):
    """The Image Gen / Edit key pool from the config (``image_job``'s own setup)."""
    try:
        from types import SimpleNamespace
        from image_job import ImageJobMixin

        shim = SimpleNamespace(config=config, append_log=log)
        ImageJobMixin._configure_media_output_key_pools(shim)
    except Exception as exc:
        log(f"⚠️ Interactive Images: could not set up the image key pool: {exc}")


def image_pool_model(config):
    """The model of the first Image Gen / Edit pool key when that pool is on (else '')."""
    if not config.get('use_inpainter_keys'):
        return ''
    for key in config.get('inpainter_keys') or ():
        if isinstance(key, dict) and key.get('model') and key.get('enabled', True):
            return str(key['model'])
    return ''


def _image_bytes_from_response(client, content):
    """``(bytes, ext)`` of the image a request returned (thread-local bytes / path, data URI or the
    ``[GENERATED_IMAGE:path]`` marker); raises when there is none."""
    try:
        tls = client._get_thread_local_client()
        data = getattr(tls, '_last_generated_image_bytes', None)
        if data:
            try:
                tls._last_generated_image_bytes = None
            except Exception:
                pass
            return bytes(data), _sniff_ext(data)
        path = getattr(tls, '_last_generated_image_path', None)
        if path and os.path.isfile(path):
            try:
                tls._last_generated_image_path = None
            except Exception:
                pass
            with open(path, 'rb') as handle:
                data = handle.read()
            return data, _sniff_ext(data)
    except Exception:
        pass
    text = str(content or '').strip()
    if text.startswith('data:image/'):
        header, _, payload = text.partition(',')
        data = base64.b64decode(payload)
        return data, _sniff_ext(data)
    match = re.search(r'\[GENERATED_IMAGE:(.*?)\]', text)
    if match and os.path.isfile(match.group(1).strip()):
        with open(match.group(1).strip(), 'rb') as handle:
            data = handle.read()
        return data, _sniff_ext(data)
    raise RuntimeError("the image model returned no image")


def _sniff_ext(data):
    head = bytes(data[:12])
    if head.startswith(b'\x89PNG'):
        return 'png'
    if head.startswith(b'\xff\xd8'):
        return 'jpg'
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'webp'
    raise RuntimeError("the image model returned data that is not a PNG, JPEG or WebP image")


def _data_url(path):
    from safe_image import open_image
    from io import BytesIO

    with open_image(path) as image:
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGBA')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def _send(client, messages, config, context, max_tokens=None):
    temperature = float(config.get('temperature', 0.3) or 0.3)
    if max_tokens is None:
        max_tokens = int(config.get('max_output_tokens', config.get('max_tokens', 8192)) or 8192)
    result = client.send(messages, temperature=temperature, max_tokens=max_tokens, context=context)
    if isinstance(result, tuple):
        return result[0]
    return result


def _messages(system_text, user_content):
    messages = []
    if str(system_text or '').strip():
        messages.append({'role': 'system', 'content': system_text})
    messages.append({'role': 'user', 'content': user_content})
    return messages


# ---------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------

def _token_counter(model):
    try:
        from direct_text_stream import count_tokens
        return lambda text: count_tokens(text, model)
    except Exception:
        return lambda text: max(1, len(str(text)) // 4)


def _planner_budget(config, settings):
    if settings.planner_max_input_tokens:
        return settings.planner_max_input_tokens
    for key in ('max_input_tokens', 'input_token_limit', 'token_limit'):
        try:
            value = int(config.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def build_planner_prompts(settings, blocks, title, names, css_classes, *, budget=0, count_tokens=None):
    """``[(prompt, chunk_blocks, images_for_chunk)]``: the whole chapter when it fits ``budget`` tokens,
    else block-aligned chunks (global paragraph numbers, each with its share of the images)."""
    count_tokens = count_tokens or (lambda text: max(1, len(text) // 4))
    values = {
        'images_per_chapter': settings.per_chapter,
        'character_names': '\n'.join(names),
        'css_classes': '\n'.join(css_classes),
        'chapter_title': title,
    }
    overhead = count_tokens(core.fill_placeholders(settings.planner_prompt, dict(values, paragraphs='')))
    chunks = core.chunk_blocks(blocks, budget, count_tokens, overhead)
    out = []
    for chunk in chunks:
        share = settings.per_chapter if len(chunks) == 1 else core.chunk_image_share(
            settings.per_chapter, len(chunk), len(blocks))
        prompt = core.fill_placeholders(settings.planner_prompt, dict(
            values, images_per_chapter=share, paragraphs=core.format_paragraphs(chunk)))
        out.append((prompt, chunk, share))
    return out


# ---------------------------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------------------------

def _book_compiled(output_dir):
    try:
        return any(name.lower().endswith(('.epub', '.pdf')) for name in os.listdir(output_dir))
    except OSError:
        return False


def prepare_stylesheet(config, output_dir, log):
    """Write (or remove) the book's interactive stylesheet for this run; returns its assignable classes."""
    settings = core.settings_from_config(config)
    if not settings.css_enabled:
        presets.remove_book_stylesheet(output_dir)
        return [], None
    name = presets.active_preset_name(config)
    css = presets.preset_css(config, name) or presets.BUILTIN_PRESETS[presets.DEFAULT_PRESET]
    _path, warnings = presets.write_book_stylesheet(output_dir, css)
    for warning in warnings:
        log(f"⚠️ Interactive styles '{name}': {warning}")
    return presets.assignable_classes(css), name


def run_interactive_images(config, output_dir, keys=None, *, log=None, stop_event=None,
                           on_progress: Optional[Callable] = None, client_factory=None, recompile=None):
    """Run the Interactive Images step on one book's output folder.

    ``keys``: output file names to work on (default: every chapter that needs a build). ``stop_event``: a
    ``threading.Event`` checked between requests. ``on_progress(key, status)`` after each chapter step.
    ``client_factory(kind, model, config, output_dir)`` (tests) returns an object with ``send``.
    ``recompile(output_dir)`` is called at the end when the book had a compiled EPUB/PDF before, something
    changed and ``interactive_images_recompile_after`` is on. Returns a :class:`RunResult`."""
    log = log or print
    stop_event = stop_event or threading.Event()
    client_factory = client_factory or _default_client_factory
    result = RunResult()
    settings = core.settings_from_config(config)
    text_model = settings.text_model or str(config.get('model') or '')
    image_model = settings.image_model or image_pool_model(config)
    wants_images = settings.per_chapter > 0
    if not text_model:
        log("❌ Interactive Images: no text model (set one, or the main translation model)")
        result.errors.append('no text model')
        return result
    if wants_images and not image_model:
        log("❌ Interactive Images: no image model set (Interactive Images settings), or set images per chapter to 0")
        result.errors.append('no image model')
        return result

    os.makedirs(core.interactive_dir(output_dir), exist_ok=True)
    path = core.progress_path(output_dir)
    css_classes, preset_name = prepare_stylesheet(config, output_dir, log)
    stylesheet = {}
    if preset_name:
        import hashlib
        css_text = presets.read_book_stylesheet(output_dir)
        stylesheet = {'preset': preset_name, 'hash': hashlib.md5(css_text.encode('utf-8')).hexdigest()}
    ipc.set_book_info(path, str(config.get('active_interactive_images_prompt_profile') or 'Default'), stylesheet)
    entries = core.load_glossary_entries(output_dir)
    name_map = core.glossary_name_map(entries)
    names = core.character_names(entries)
    refs = core.load_references(output_dir)
    max_refs = core.effective_reference_count(image_model, settings.max_references)
    if wants_images and core.reference_limit(image_model) == 0:
        log(f"ℹ️ Interactive Images: {core.NO_REFERENCE_WARNING}")
    if wants_images:
        configure_image_key_pool(config, log)
    count_tokens = _token_counter(text_model)
    budget = _planner_budget(config, settings)

    model = ipc.open_progress(output_dir)
    chapters = model.chapters
    if keys is None:
        wanted = set(ipc.pending_keys(model))
    else:
        wanted = {os.path.basename(str(key)) for key in keys}
    work = [chapter for chapter in chapters if chapter.key in wanted]
    log(f"🖼 Interactive Images: {len(work)} chapter(s), {settings.per_chapter} image(s) per chapter, "
        f"text model {text_model}" + (f", image model {image_model}" if wants_images else ''))

    text_client = None
    image_client = None
    for chapter in work:
        if stop_event.is_set():
            result.stopped = True
            break
        result.processed += 1
        if not ipc.translation_completed(chapter):
            ipc.set_status(path, chapter.key, 'skipped', f"translation status is {chapter.status or 'unknown'}")
            result.skipped += 1
            _notify(on_progress, chapter.key, 'skipped')
            continue
        try:
            if text_client is None:
                text_client = client_factory('text', text_model, config, output_dir)
            if wants_images and image_client is None:
                image_client = client_factory('image', image_model, config, output_dir)
            status = _run_chapter(config, settings, output_dir, path, chapter, text_client, image_client,
                                  text_model, image_model, css_classes, names, name_map, refs, max_refs,
                                  count_tokens, budget, stop_event, log, on_progress)
        except _Stop:
            ipc.set_status(path, chapter.key, 'pending', 'stopped')
            result.stopped = True
            _notify(on_progress, chapter.key, 'pending')
            break
        except Exception as exc:
            log(f"❌ Interactive Images {chapter.key}: {exc}")
            if not isinstance(exc, (core.PlanError, RuntimeError)):
                log(traceback.format_exc())
            ipc.set_status(path, chapter.key, 'failed', str(exc))
            result.failed += 1
            result.errors.append(f"{chapter.key}: {exc}")
            _notify(on_progress, chapter.key, 'failed')
            continue
        if status in ('completed', 'css_only', 'partial'):
            result.completed += 1
            result.changed_files.append(chapter.key)
        elif status == 'rejected':
            result.rejected += 1
        elif status == 'skipped':
            result.skipped += 1
        else:
            result.failed += 1
        _notify(on_progress, chapter.key, status)

    try:
        result.images = ipc.image_stats(ipc.open_progress(output_dir)).get('images', 0)
    except Exception:
        pass
    log(f"🖼 Interactive Images done: {result.completed} built, {result.failed} failed, {result.rejected} rejected, "
        f"{result.skipped} skipped" + (" (stopped)" if result.stopped else ''))
    if (recompile is not None and result.changed_files and settings.recompile_after and not result.stopped
            and _book_compiled(output_dir)):
        try:
            log("📦 Interactive Images: recompiling the book")
            recompile(output_dir)
        except Exception as exc:
            log(f"⚠️ Interactive Images: recompile failed: {exc}")
    return result


def _notify(callback, key, status):
    if callback is not None:
        try:
            callback(key, status)
        except Exception:
            pass


def _check_stop(stop_event):
    if stop_event.is_set():
        raise _Stop()


def _run_chapter(config, settings, output_dir, path, chapter, text_client, image_client, text_model, image_model,
                 css_classes, names, name_map, refs, max_refs, count_tokens, budget, stop_event, log, on_progress):
    with open(chapter.path, 'r', encoding='utf-8') as handle:
        original = handle.read()
    base = core.strip_injections(original)
    ok, hint = core.check_text_unchanged(original, base)
    if not ok:
        raise RuntimeError(f"could not remove the earlier insertions cleanly ({hint})")
    blocks = core.number_blocks(base)
    if not blocks:
        ipc.set_status(path, chapter.key, 'skipped', 'no text paragraphs')
        return 'skipped'
    ipc.set_status(path, chapter.key, 'in_progress')
    _notify(on_progress, chapter.key, 'in_progress')

    # 1. plan
    title = core.chapter_title(base, chapter.title or chapter.key)
    plans = []
    for prompt, chunk, share in build_planner_prompts(settings, blocks, title, names, css_classes,
                                                      budget=budget, count_tokens=count_tokens):
        _check_stop(stop_event)
        reply = _send(text_client, _messages('', prompt), config, PLAN_CONTEXT)
        plans.append(core.parse_plan(reply, len(blocks), css_classes, share, glossary_names=name_map))
    plan = core.merge_plans(plans, settings.per_chapter)
    for warning in plan.warnings:
        log(f"ℹ️ {chapter.key}: {warning}")

    # 2. place it (placeholder image sources)
    placed_images, unplaced = [], []
    for item in plan.images:
        block = core.locate_block(blocks, item.after_paragraph, item.quote)
        if block is None:
            item.status = 'unplaced'
            unplaced.append(item)
            continue
        item.after_paragraph = block.n
        placed_images.append((block, item))
    placed_images.sort(key=lambda pair: pair[0].n)
    placed_classes = []
    for item in plan.classes:
        block = core.locate_block(blocks, item.paragraph, item.quote)
        if block is None:
            log(f"ℹ️ {chapter.key}: class '{item.css_class}' could not be placed (paragraph {item.paragraph})")
            continue
        placed_classes.append((block, item.css_class))
    pending = [(block, _PENDING_SRC.format(k=k)) for k, (block, _item) in enumerate(placed_images, 1)]
    draft = core.apply_plan(base, blocks, pending, placed_classes, book_class=settings.css_enabled)
    ok, hint = core.check_text_unchanged(original, draft)
    if not ok:
        ipc.set_entry(path, chapter.key, _entry(chapter, 'rejected', original, text_model, image_model, [], [],
                                                f"plan would change the text: {hint}"))
        log(f"🚫 {chapter.key}: rejected, the plan would change the chapter text ({hint})")
        return 'rejected'

    # 3. images
    records = []
    final = draft
    folder = core.interactive_dir(output_dir)
    kept_files = set()
    for k, (block, item) in enumerate(placed_images, 1):
        placeholder = _PENDING_SRC.format(k=k)
        record = {'file': '', 'after_paragraph': block.n, 'quote': item.quote, 'characters': item.characters,
                  'prompt': item.prompt, 'refs': [], 'status': 'failed', 'error': ''}
        try:
            _check_stop(stop_event)
            scene = [item.matched.get(name, name) for name in item.characters]
            picked = core.select_references(refs, scene, max_refs)
            record['refs'] = [rel for _name, rel in picked]
            user_text = core.fill_placeholders(settings.image_prompt, {
                'scene_prompt': item.prompt,
                'characters': ', '.join(scene),
                'reference_list': core.reference_list_text(picked, settings.reference_line),
            })
            content = [{'type': 'text', 'text': user_text}]
            for _name, rel in picked:
                full = os.path.join(folder, rel)
                if os.path.isfile(full):
                    content.append({'type': 'image_url', 'image_url': {'url': _data_url(full)}})
            reply = _send(image_client, _messages(settings.image_system_prompt, content), config, IMAGE_CONTEXT)
            data, ext = _image_bytes_from_response(image_client, reply)
            name = core.image_filename(chapter.number, k, ext)
            target = os.path.join(folder, name)
            tmp = target + '.tmp'
            with open(tmp, 'wb') as handle:
                handle.write(data)
            os.replace(tmp, target)
            for other in core.chapter_images_on_disk(output_dir, chapter.number):
                if other != name and other.rsplit('.', 1)[0] == name.rsplit('.', 1)[0]:
                    _remove(os.path.join(folder, other))  # same k, older extension
            kept_files.add(name)
            final = core.replace_image_src(final, placeholder, core.image_src(name))
            record.update(file=name, status='completed')
            log(f"🖼 {chapter.key}: image {k} saved as {name}")
        except _Stop:
            raise
        except Exception as exc:
            record['error'] = str(exc)
            final = core.remove_figure(final, placeholder)
            log(f"⚠️ {chapter.key}: image {k} failed: {exc}")
        records.append(record)
    for item in unplaced:
        records.append({'file': '', 'after_paragraph': item.after_paragraph, 'quote': item.quote,
                        'characters': item.characters, 'prompt': item.prompt, 'refs': [], 'status': 'unplaced',
                        'error': 'paragraph not found'})
    done = sum(1 for record in records if record['status'] == 'completed')
    if placed_images and not done and not placed_classes:
        # nothing to show: the chapter file and an earlier build stay as they were
        error = '; '.join(record['error'] for record in records if record['error'])[:500]
        ipc.set_status(path, chapter.key, 'failed', error or 'no image was generated')
        log(f"❌ {chapter.key}: no image was generated; the chapter is unchanged")
        return 'failed'
    # images of an earlier run that this run does not have any more
    for other in core.chapter_images_on_disk(output_dir, chapter.number):
        if other not in kept_files:
            _remove(os.path.join(folder, other))
    if placed_images and done < len(placed_images):
        status, error = 'partial', ''
    elif not placed_images and not plan.images:
        status, error = 'css_only', ''
    elif not placed_images:
        status, error = 'partial', 'no image could be placed'
    else:
        status, error = 'completed', ''

    ok, hint = core.check_text_unchanged(original, final)
    if not ok:
        for name in kept_files:
            _remove(os.path.join(folder, name))
        ipc.set_entry(path, chapter.key, _entry(chapter, 'rejected', original, text_model, image_model, records,
                                                [], f"text changed: {hint}"))
        log(f"🚫 {chapter.key}: rejected ({hint})")
        return 'rejected'
    if final != original:
        ipc.write_text_atomic(chapter.path, final)
    classes = [{'paragraph': block.n, 'class': name} for block, name in placed_classes] if status != 'failed' else []
    ipc.set_entry(path, chapter.key, _entry(chapter, status, final, text_model, image_model, records, classes, error))
    log(f"✅ {chapter.key}: {done} image(s), {len(classes)} class(es) ({status})")
    return status


def _entry(chapter, status, html, text_model, image_model, images, classes, error):
    return {
        'chapter': chapter.number,
        'status': status,
        'source_hash': core.visible_text_hash(html),
        'built_at': ipc.now_iso(),
        'text_model': text_model,
        'image_model': image_model,
        'images': images,
        'classes': classes,
        'error': error,
    }


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------------------------
# After a translation run (translation_pipeline's post-run block; desktop and mobile)
# ---------------------------------------------------------------------------------------------

def owner_recompiler(owner):
    """``recompile(folder)`` over an owner's compile methods (``text_jobs``: desktop and the mobile
    HeadlessOwner): the EPUB when the folder has one, else the PDF; None when the owner has neither."""
    compile_epub = getattr(owner, '_run_epub_compile', None)
    compile_pdf = getattr(owner, '_run_pdf_compile', None)
    if not callable(compile_epub) and not callable(compile_pdf):
        return None

    def recompile(folder):
        try:
            names = os.listdir(folder)
        except OSError:
            names = []
        if any(n.lower().endswith('.epub') for n in names) and callable(compile_epub):
            compile_epub(folder)
        elif any(n.lower().endswith('.pdf') for n in names) and callable(compile_pdf):
            compile_pdf(folder)
    return recompile


class _OwnerStop:
    """A stop event that follows the owner's Stop button."""

    def __init__(self, owner):
        self.owner = owner
        self._event = threading.Event()

    def is_set(self):
        return self._event.is_set() or bool(getattr(self.owner, 'stop_requested', False))

    def set(self):
        self._event.set()


def run_output_dir(owner, source):
    """The output folder the run just wrote for ``source``: the translation run's own resolver
    (``run_env._resolve_translation_output_dir``, desktop and the mobile HeadlessOwner), else the Progress
    Manager's (``progress_core.resolve_output_dir``)."""
    resolvers = [getattr(owner, '_resolve_translation_output_dir', None)]
    try:
        from progress_core import resolve_output_dir
        resolvers.append(lambda path: resolve_output_dir(owner, path))
    except Exception:
        pass
    for resolve in resolvers:
        if not callable(resolve):
            continue
        try:
            folder = resolve(source)
        except Exception:
            folder = None
        if folder and os.path.isdir(folder):
            return folder
    return None


def run_after_translation(owner, completed):
    """Run the step on the books of a translation run that ended normally, when
    ``interactive_images_auto_after_translation`` is on. Never raises: a failure is logged."""
    log = getattr(owner, 'append_log', None) or print
    try:
        config = getattr(owner, 'config', None) or {}
        if not completed or not core.settings_from_config(config).auto_after_translation:
            return []
        results = []
        seen = set()
        for source in list(getattr(owner, 'selected_files', None) or ()):
            if not str(source).lower().endswith(('.epub', '.txt', '.pdf', '.html', '.htm', '.docx', '.md')):
                continue
            folder = run_output_dir(owner, source)
            if not folder or not os.path.isfile(os.path.join(folder, 'translation_progress.json')):
                log(f"ℹ️ Interactive Images after translation: no translation output found for "
                    f"{os.path.basename(str(source))}")
                continue
            key = os.path.normcase(os.path.abspath(folder))
            if key in seen:
                continue
            seen.add(key)
            results.append(run_interactive_images(
                config, folder, log=log, stop_event=_OwnerStop(owner), recompile=owner_recompiler(owner)))
        return results
    except Exception as exc:
        log(f"⚠️ Interactive Images after translation failed: {exc}")
        return []


# ---------------------------------------------------------------------------------------------
# A stylesheet from the user's request (becomes a user preset)
# ---------------------------------------------------------------------------------------------

def generate_stylesheet(config, user_request, *, client_factory=None, output_dir=None):
    """``(css, warnings)`` from the profile's CSS prompt with ``{user_request}`` and ``{css_classes}`` (the
    class vocabulary), sent to the text model; the first css code block, scoped."""
    settings = core.settings_from_config(config)
    model = settings.text_model or str(config.get('model') or '')
    if not model:
        raise RuntimeError("no text model")
    client_factory = client_factory or _default_client_factory
    prompt = core.fill_placeholders(settings.css_prompt, {
        'user_request': str(user_request or '').strip(),
        'css_classes': '\n'.join(presets.CLASS_VOCABULARY),
    })
    import tempfile
    client = client_factory('text', model, config, output_dir or tempfile.gettempdir())
    reply = _send(client, _messages('', prompt), config, CSS_CONTEXT)
    return presets.scope_css(presets.extract_css_block(reply))


__all__ = ['RunResult', 'build_planner_prompts', 'configure_image_key_pool', 'generate_stylesheet',
           'image_pool_model', 'owner_recompiler', 'prepare_stylesheet', 'run_after_translation',
           'run_interactive_images']
