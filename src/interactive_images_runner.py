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
import contextlib
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
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
    no_image_model: str = ''
    changed_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class _Stop(Exception):
    pass


@dataclass
class ImageModelChoice:
    """The model that draws and why: ``source`` is ``'the Interactive Images image model'``, ``'the Image Gen / Edit
    pool'`` or ``'the text model'``; ``capability`` is ``image_capability`` of a text model used for pictures."""
    model: str = ''
    source: str = ''
    capability: str = ''

    @property
    def probe(self):
        """The first image request decides (the text model is of unknown capability)."""
        return self.source == 'the text model' and self.capability == 'unknown'


def resolve_image_model(config, settings, text_model):
    """1) the Interactive Images image model, 2) the Image Gen / Edit pool model, 3) the text model when it is not
    known to be text-only. ``model`` is '' when none can draw."""
    if settings.image_model:
        return ImageModelChoice(settings.image_model, 'the Interactive Images image model')
    pooled = image_pool_model(config)
    if pooled:
        return ImageModelChoice(pooled, 'the Image Gen / Edit pool')
    capability = core.image_capability(text_model)
    if capability == 'text_only':
        return ImageModelChoice('', 'the text model', capability)
    return ImageModelChoice(text_model, 'the text model', capability)


class _ImageProbe:
    """A text model of unknown capability draws the pictures: its first image request is the probe. Other image
    requests wait for the verdict; when no image comes back the run stops (no retries, no more planner requests)."""

    def __init__(self, model, stop_event, result, log):
        self.model = model
        self.stop_event = stop_event
        self.result = result
        self.log = log
        self.verdict = None
        self._lock = threading.Lock()

    def run(self, send):
        if self.verdict is None:
            with self._lock:
                if self.verdict is None:
                    try:
                        out = send()
                    except _Stop:
                        raise
                    except Exception as exc:
                        self.verdict = False
                        message = (f"the text model {self.model} was tried for pictures (it is not known to draw "
                                   f"them) and its first image request returned no image ({exc}). Set an Image "
                                   f"model (Interactive Images settings or the Image Gen / Edit pool), or set "
                                   f"images per chapter to 0.")
                        self.result.errors.append('no image model')
                        self.result.no_image_model = message
                        self.log(f"❌ Interactive Images: {message}")
                        self.stop_event.set()
                        raise _Stop() from None
                    self.verdict = True
                    self.log(f"ℹ️ Interactive Images: {self.model} returned a picture; it keeps drawing")
                    return out
        if self.verdict is False:
            raise _Stop()
        return send()


# ---------------------------------------------------------------------------------------------
# Batch mode (the translation's own settings: Batch Translation, batch size, threading delay, API call delay)
# ---------------------------------------------------------------------------------------------

#: The most chapters / requests in flight at once, whatever the batch size says.
MAX_BATCH_WORKERS = 16


@dataclass
class BatchSettings:
    """The translation's batch settings as this step reads them. ``enabled`` with ``size`` > 1 runs chapters (and a
    chapter's image requests) in parallel, at most ``workers`` requests in flight; otherwise the step is sequential
    exactly as before. The delay strings are the translation's env values (``SEND_INTERVAL_SECONDS`` and
    ``SEND_INTERVAL_MAX_SECONDS``, ``THREAD_SUBMISSION_DELAY_SECONDS``)."""
    enabled: bool = False
    size: int = 1
    send_interval: str = ''
    send_interval_max: str = ''
    thread_delay: str = ''

    @property
    def workers(self):
        if not self.enabled:
            return 1
        try:
            size = int(self.size)
        except (TypeError, ValueError):
            size = 1
        return max(1, min(size, MAX_BATCH_WORKERS))

    @property
    def parallel(self):
        return self.workers > 1

    def thread_delay_seconds(self):
        try:
            return max(0.0, float(self.thread_delay))
        except (TypeError, ValueError):
            return 0.0


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def batch_settings_from_owner(owner):
    """The translation's batch settings of the app (desktop window or the mobile HeadlessOwner): the live
    ``batch_translation_var`` / ``batch_size_var`` / API call delay / threading delay fields, else the saved config."""
    config = getattr(owner, 'config', None)
    config = config if isinstance(config, dict) else {}
    # a real owner always has the var; without one (and no saved value) there is no batch setting to follow
    enabled = _truthy(getattr(owner, 'batch_translation_var', config.get('batch_translation', False)))
    raw_size = getattr(owner, 'batch_size_var', None)
    if raw_size in (None, ''):
        raw_size = config.get('batch_size')
    try:
        size = int(str(raw_size).strip())
    except (TypeError, ValueError):
        from settings_rules import default_batch_size
        size = default_batch_size(config.get('model'))
    try:
        from settings_rules import delay_env, delay_range_text, owner_delay_env
        if getattr(owner, 'delay_entry', None) is not None:
            interval, interval_max = owner_delay_env(owner, '2')
        else:
            interval, interval_max = delay_env(delay_range_text(config))
    except Exception:
        interval, interval_max = '2', ''
    widget = getattr(owner, 'thread_delay_entry', None)
    thread_delay = ''
    if widget is not None:
        try:
            thread_delay = str(widget.text()).strip()
        except Exception:
            thread_delay = ''
    if not thread_delay:
        thread_delay = str(config.get('thread_submission_delay', '0.0001')).strip()
    return BatchSettings(enabled=enabled, size=max(1, size), send_interval=str(interval),
                         send_interval_max=str(interval_max or ''), thread_delay=thread_delay or '0.0001')


@contextlib.contextmanager
def _batch_environment(batch):
    """The translation's env (``BATCH_TRANSLATION``, ``BATCH_SIZE``, the two delays) for a parallel run, put back after:
    ``UnifiedClient`` reads them (batch mode lets requests overlap and keeps one request's failure from cancelling its
    siblings; the API call delay staggers the request starts)."""
    values = {'BATCH_TRANSLATION': '1', 'BATCH_SIZE': str(batch.workers),
              'SEND_INTERVAL_SECONDS': batch.send_interval, 'SEND_INTERVAL_MAX_SECONDS': batch.send_interval_max,
              'THREAD_SUBMISSION_DELAY_SECONDS': batch.thread_delay}
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if name in ('SEND_INTERVAL_SECONDS', 'THREAD_SUBMISSION_DELAY_SECONDS') and value == '':
                continue   # no value from the owner: whatever the process already has stays
            os.environ[name] = value
        yield
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old


class _Clients:
    """The step's clients. Sequential: one text client and one image client, made on first use, as before. Parallel:
    a client is checked out for one request at a time (the image aspect ratio and the forced-image flags live on the
    client, so two threads never share one); idle clients are reused, so at most ``workers`` exist per kind."""

    def __init__(self, factory, models, config, output_dir):
        self._factory = factory
        self._models = models
        self._config = config
        self._output_dir = output_dir
        self._idle = {'text': [], 'image': []}
        self._made = {'text': 0, 'image': 0}
        self._lock = threading.Lock()

    def warm(self, kind):
        """Make the first client of a kind now, so a client that cannot be made fails the chapter before any request
        (as it always has), not one image at a time."""
        with self._lock:
            if not self._made[kind]:
                self._idle[kind].append(self._factory(kind, self._models[kind], self._config, self._output_dir))
                self._made[kind] += 1

    @contextlib.contextmanager
    def checkout(self, kind):
        with self._lock:
            client = self._idle[kind].pop() if self._idle[kind] else None
        if client is None:
            client = self._factory(kind, self._models[kind], self._config, self._output_dir)
            with self._lock:
                self._made[kind] += 1
        try:
            yield client
        finally:
            with self._lock:
                self._idle[kind].append(client)


class _Dispatcher:
    """Bounds and paces the requests of a parallel run: at most ``workers`` in flight, request starts at least the
    threading delay apart (the next-slot reservation ``UnifiedClient._apply_thread_submission_delay`` uses, shared by
    every client here). Sequential runs (``workers`` 1) pass straight through. Both waits end with ``_Stop``."""

    def __init__(self, batch, stop_event):
        self.stop_event = stop_event
        self.workers = batch.workers if batch is not None else 1
        self.delay = batch.thread_delay_seconds() if batch is not None else 0.0
        self._slots = threading.BoundedSemaphore(self.workers) if self.workers > 1 else None
        self._lock = threading.Lock()
        self._next = 0.0

    @contextlib.contextmanager
    def request(self):
        if self._slots is None:
            yield
            return
        while not self._slots.acquire(timeout=0.1):
            _check_stop(self.stop_event)
        try:
            _check_stop(self.stop_event)
            self._space_start()
            yield
        finally:
            self._slots.release()

    def _space_start(self):
        if self.delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.delay
        while True:
            remaining = start - time.monotonic()
            if remaining <= 0:
                return
            _check_stop(self.stop_event)
            time.sleep(min(0.1, remaining))


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
        # the aspect ratio is set per image (``_set_image_aspect``: the fixed ratio, or one picked by Auto)
        client._suppress_custom_image_edit_endpoint = True
    else:
        # a text request (planner, CSS): the main tab's Output mode (image / video) must not turn it into an
        # image request, which would also send it to the Image Gen / Edit key pool
        client._text_output_only = True
    return client


def _set_image_aspect(client, ratio):
    """The ratio the next image request asks for (the API parameter ``UnifiedClient`` reads as
    ``_forced_image_output_aspect_ratio``); '' = none, the model decides."""
    try:
        client._forced_image_output_aspect_ratio = ratio or None
    except Exception:
        pass


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
    ``[GENERATED_IMAGE:path]`` marker); raises when there is none.

    The client also saves its own copy of the reply (``image_generation_response_*.png`` in the book's
    output folder); the step keeps only its ``Glossarion_chNNN_k`` file, so that copy is removed."""
    text = str(content or '').strip()
    match = re.search(r'\[GENERATED_IMAGE:(.*?)\]', text)
    saved = {match.group(1).strip()} if match else set()
    found = None
    try:
        tls = client._get_thread_local_client()
        path = getattr(tls, '_last_generated_image_path', None)
        if path:
            saved.add(str(path))
            try:
                tls._last_generated_image_path = None
            except Exception:
                pass
        data = getattr(tls, '_last_generated_image_bytes', None)
        if data:
            try:
                tls._last_generated_image_bytes = None
            except Exception:
                pass
            found = bytes(data)
        elif path and os.path.isfile(path):
            with open(path, 'rb') as handle:
                found = handle.read()
    except Exception:
        pass
    if found is None and text.startswith('data:image/'):
        header, _, payload = text.partition(',')
        found = base64.b64decode(payload)
    if found is None and match and os.path.isfile(match.group(1).strip()):
        with open(match.group(1).strip(), 'rb') as handle:
            found = handle.read()
    if found is None:
        raise RuntimeError("the image model returned no image")
    ext = _sniff_ext(found)
    for path in saved:
        _remove(path)
    return found, ext


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
        'image_subjects': core.subjects_text(settings),
        'css_classes': '\n'.join(css_classes),
        'chapter_title': title,
    }
    drop = core.LINE_DROP_PLACEHOLDERS
    overhead = count_tokens(core.fill_placeholders(settings.planner_prompt, dict(values, paragraphs=''), drop))
    chunks = core.chunk_blocks(blocks, budget, count_tokens, overhead)
    out = []
    for chunk in chunks:
        share = settings.per_chapter if len(chunks) == 1 else core.chunk_image_share(
            settings.per_chapter, len(chunk), len(blocks))
        prompt = core.fill_placeholders(settings.planner_prompt, dict(
            values, images_per_chapter=share, paragraphs=core.format_paragraphs(chunk)), drop)
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
                           on_progress: Optional[Callable] = None, client_factory=None, recompile=None,
                           batch: Optional[BatchSettings] = None, glossary_path: Optional[str] = None):
    """Run the Interactive Images step on one book's output folder.

    ``glossary_path``: the glossary translation uses for this book (``core.find_book_glossary(output_dir,
    owner)``, resolved by the caller on its own thread); '' = translation uses none; None = the plain lookup.

    ``batch``: the translation's batch settings (:func:`batch_settings_from_owner`); None or Batch Translation off
    (or batch size 1) = one request at a time. On, up to ``batch.workers`` chapters run at once and a chapter's image
    requests overlap, with at most that many requests in flight.

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
    choice = resolve_image_model(config, settings, text_model) if text_model else ImageModelChoice()
    image_model = choice.model
    wants_images = settings.per_chapter > 0
    if not text_model:
        log("❌ Interactive Images: no text model (set one, or the main translation model)")
        result.errors.append('no text model')
        return result
    if wants_images and not image_model:
        result.no_image_model = (f"no image-capable model found: no Interactive Images image model, no Image Gen / Edit "
                                 f"pool model, and the text model {text_model} is text-only. Set an image model, or "
                                 f"set images per chapter to 0")
        log(f"❌ Interactive Images: {result.no_image_model}")
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
    glossary = core.find_book_glossary(output_dir) if glossary_path is None else (glossary_path or None)
    entries = core.load_glossary_entries(output_dir, glossary) if glossary else []
    name_map = core.glossary_name_map(entries)
    names = core.character_names(entries)
    tracker = core.load_gender_tracker(output_dir, glossary) if settings.include_gender and glossary else {}
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
        f"text model {text_model}" + (f", image model {image_model} ({choice.source})" if wants_images else ''))

    # batch mode (the translation's settings): chapters and a chapter's image requests overlap; off = one at a time
    if batch is not None and not batch.parallel:
        batch = None
    probe = _ImageProbe(image_model, stop_event, result, log) if wants_images and choice.probe else None
    if probe is not None:
        log(f"\u2139\ufe0f Interactive Images: the text model {image_model} draws the pictures; it is not known "
            f"whether it can, so its first image request is the test")
    clients = _Clients(client_factory, {'text': text_model, 'image': image_model}, config, output_dir)
    dispatcher = _Dispatcher(batch, stop_event)
    result_lock = threading.Lock()
    notify_lock = threading.Lock()

    def handle(chapter, chapter_log):
        """One chapter, start to finish; a failure stays this chapter's. True when the run was stopped."""
        with result_lock:
            result.processed += 1
        if not ipc.translation_completed(chapter):
            ipc.set_status(path, chapter.key, 'skipped', f"translation status is {chapter.status or 'unknown'}")
            with result_lock:
                result.skipped += 1
            _notify(on_progress, chapter.key, 'skipped', notify_lock)
            return False
        try:
            clients.warm('text')
            if wants_images:
                clients.warm('image')
            status = _run_chapter(config, settings, output_dir, path, chapter, clients, dispatcher,
                                  text_model, image_model, css_classes, (names, entries, tracker), name_map, refs,
                                  max_refs, count_tokens, budget, stop_event, chapter_log, on_progress, notify_lock,
                                  batch, probe)
        except _Stop:
            ipc.set_status(path, chapter.key, 'pending', 'stopped')
            with result_lock:
                result.stopped = True
            _notify(on_progress, chapter.key, 'pending', notify_lock)
            return True
        except Exception as exc:
            chapter_log(f"❌ Interactive Images {chapter.key}: {exc}")
            if not isinstance(exc, (core.PlanError, RuntimeError)):
                chapter_log(traceback.format_exc())
            ipc.set_status(path, chapter.key, 'failed', str(exc))
            with result_lock:
                result.failed += 1
                result.errors.append(f"{chapter.key}: {exc}")
            _notify(on_progress, chapter.key, 'failed', notify_lock)
            return False
        with result_lock:
            if status in ('completed', 'css_only', 'partial'):
                result.completed += 1
                result.changed_files.append(chapter.key)
            elif status == 'rejected':
                result.rejected += 1
            elif status == 'skipped':
                result.skipped += 1
            else:
                result.failed += 1
        _notify(on_progress, chapter.key, status, notify_lock)
        return False

    if batch is None:
        for chapter in work:
            if stop_event.is_set():
                result.stopped = True
                break
            if handle(chapter, log):
                break
    else:
        log(f"⚡ Interactive Images batch mode: up to {batch.workers} requests in parallel")

        def worker(chapter):
            if stop_event.is_set():
                with result_lock:
                    result.stopped = True
                return
            tag = threading.current_thread().name   # every line of this chapter carries its worker's tag
            handle(chapter, lambda message: log(f"[{tag}] {message}"))

        with _batch_environment(batch), ThreadPoolExecutor(
                max_workers=batch.workers, thread_name_prefix='ii-worker') as pool:
            for future in [pool.submit(worker, chapter) for chapter in work]:
                future.result()
        order = {chapter.key: n for n, chapter in enumerate(work)}
        result.changed_files.sort(key=lambda key: order.get(key, 0))

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


def _notify(callback, key, status, lock=None):
    if callback is not None:
        try:
            if lock is None:
                callback(key, status)
            else:
                with lock:   # progress callbacks never run at the same time
                    callback(key, status)
        except Exception:
            pass


def _check_stop(stop_event):
    if stop_event.is_set():
        raise _Stop()


def _run_chapter(config, settings, output_dir, path, chapter, clients, dispatcher, text_model, image_model,
                 css_classes, characters, name_map, refs, max_refs, count_tokens, budget, stop_event, log,
                 on_progress, notify_lock=None, batch=None, probe=None):
    names, entries, tracker = characters
    # confident genders at this chapter (glossary + gender tracker), written through the editable placeholder text
    genders = core.character_genders(entries, tracker, chapter.number) if settings.include_gender else {}
    planner_names = [core.format_character(name, genders.get(name), settings) for name in names]
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
    _notify(on_progress, chapter.key, 'in_progress', notify_lock)

    # 1. plan
    title = core.chapter_title(base, chapter.title or chapter.key)
    plans = []
    for prompt, chunk, share in build_planner_prompts(settings, blocks, title, planner_names, css_classes,
                                                      budget=budget, count_tokens=count_tokens):
        _check_stop(stop_event)
        with dispatcher.request(), clients.checkout('text') as text_client:
            reply = _send(text_client, _messages('', prompt), config, PLAN_CONTEXT)
        plans.append(core.parse_plan(reply, len(blocks), css_classes, share, glossary_names=name_map))
    if settings.subjects == 'backgrounds':
        for plan in plans:   # Backgrounds only: a scene has no characters, whatever the model listed
            for item in plan.images:
                item.characters, item.matched = [], {}
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

    def make_image(k, block, item):
        """One image: the request and its file. Returns the progress record (``file`` set when it succeeded)."""
        record = {'file': '', 'after_paragraph': block.n, 'quote': item.quote, 'characters': item.characters,
                  'prompt': item.prompt, 'refs': [], 'status': 'failed', 'error': ''}
        try:
            _check_stop(stop_event)
            scene = [item.matched.get(name, name) for name in item.characters]
            record['scene'] = scene
            picked = core.select_references(refs, scene, max_refs)
            record['refs'] = [rel for _name, rel in picked]
            ratio = core.pick_aspect_ratio(settings, chapter.number, k)
            record['aspect_ratio'] = ratio
            user_text = core.fill_placeholders(settings.image_prompt, {
                'scene_prompt': item.prompt,
                'characters': ', '.join(core.format_character(name, genders.get(name), settings) for name in scene),
                'reference_list': core.reference_list_text(picked, settings.reference_line,
                                                           core.placeholder_text(settings, 'style_reference')),
            }, core.LINE_DROP_PLACEHOLDERS)
            content = [{'type': 'text', 'text': user_text}]
            for _name, rel in picked:
                full = os.path.join(folder, rel)
                if os.path.isfile(full):
                    content.append({'type': 'image_url', 'image_url': {'url': _data_url(full)}})
            def request():
                with dispatcher.request(), clients.checkout('image') as image_client:
                    # a client serves one request at a time: its aspect ratio and the reply bytes are its own
                    _set_image_aspect(image_client, ratio)
                    reply = _send(image_client, _messages(settings.image_system_prompt, content), config,
                                  IMAGE_CONTEXT)
                    return _image_bytes_from_response(image_client, reply)

            data, ext = probe.run(request) if probe is not None else request()
            name = core.image_filename(chapter.number, k, ext)
            target = os.path.join(folder, name)
            tmp = target + '.tmp'
            with open(tmp, 'wb') as handle:
                handle.write(data)
            os.replace(tmp, target)
            for other in core.chapter_images_on_disk(output_dir, chapter.number):
                if other != name and other.rsplit('.', 1)[0] == name.rsplit('.', 1)[0]:
                    _remove(os.path.join(folder, other))  # same k, older extension
            record.update(file=name, status='completed')
            log(f"🖼 {chapter.key}: image {k} saved as {name}")
        except _Stop:
            raise
        except Exception as exc:
            record['error'] = str(exc)
            log(f"⚠️ {chapter.key}: image {k} failed: {exc}")
        return record

    jobs = [(k, block, item) for k, (block, item) in enumerate(placed_images, 1)]
    if batch is not None and len(jobs) > 1:
        # the chapter's images overlap; the dispatcher keeps the requests of the whole run within the batch size
        outcomes = []
        with ThreadPoolExecutor(max_workers=min(len(jobs), batch.workers), thread_name_prefix='ii-image') as pool:
            futures = [pool.submit(make_image, *job) for job in jobs]
            for future in futures:
                try:
                    outcomes.append(future.result())
                except _Stop as stop:
                    outcomes.append(stop)
        if any(isinstance(outcome, _Stop) for outcome in outcomes):
            raise _Stop()
    else:
        outcomes = [make_image(*job) for job in jobs]
    for (k, _block, _item), record in zip(jobs, outcomes):
        placeholder = _PENDING_SRC.format(k=k)
        if record['file']:
            kept_files.add(record['file'])
            final = core.replace_image_src(final, placeholder, core.image_src(record['file']))
        else:
            final = core.remove_figure(final, placeholder)
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
                config, folder, log=log, stop_event=_OwnerStop(owner), recompile=owner_recompiler(owner),
                batch=batch_settings_from_owner(owner)))
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


__all__ = ['BatchSettings', 'RunResult', 'batch_settings_from_owner', 'build_planner_prompts', 'configure_image_key_pool', 'generate_stylesheet',
           'image_pool_model', 'owner_recompiler', 'prepare_stylesheet', 'run_after_translation',
           'run_interactive_images']
