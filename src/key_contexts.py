"""Request routes shared by key selection and the multi-key manager UI."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
import threading


CONTEXT_LABELS = {
    'translation': '🌐 Translation',
    'review': '📝 Review',
    'glossary': '📖 Glossary',
    'glossary_refinement': '📚 Glossary refinement',
    'refinement': '✨ Translation refinement',
    'book_title': '📕 Book title',
    'metadata': '🏷️ Metadata',
    'batch_toc_translation': '📑 Table of contents',
    'batch_header_translation': '🔖 Chapter headers',
    'summary': '🧠 Rolling summary',
    'qa_truncation': '🔍 AI truncation detection',
    'truncation': '✂️ Vision truncation scan',
    'image_scan': '🔎 Image scan',
    'image_ocr': '🔤 Image OCR',
    'vision_ocr': '👁️ Vision OCR',
    'manga_ocr': '💬 Manga OCR',
    'image_translation': '🖼️ Image translation',
    'image': '🎨 Image',
    'image_generation': '🖌️ Image generation / rendering',
    'inpainter': '🩹 Inpainter',
    'image_edit': '🛠️ Image edit',
    'custom_image_edit': '🎭 Custom image edit',
    'manga_image_edit': '🗯️ Manga image edit',
    'tts': '🔊 Audio / TTS',
    'other': '🧩 Other contexts',
}

_CHAT_CONTEXTS = tuple(c for c in CONTEXT_LABELS if c != 'tts')
POOL_CONTEXTS = {
    # Without a dedicated audio pool, speech requests also rotate main keys.
    'main': tuple(CONTEXT_LABELS),
    'fallback': _CHAT_CONTEXTS,
    'glossary': ('glossary', 'glossary_refinement'),
    'glossary_refinement': ('refinement', 'glossary_refinement'),
    'metadata': ('book_title', 'metadata', 'batch_toc_translation', 'batch_header_translation'),
    'qa_scan': ('truncation', 'image_scan', 'image_ocr', 'vision_ocr', 'manga_ocr', 'image_translation'),
    'ai_truncation_detection': ('qa_truncation',),
    'rolling_summary': ('summary',),
    # A truncation retry retains the context of the original request.
    'truncation_retry': _CHAT_CONTEXTS,
    'inpainter': ('image', 'image_translation', 'image_generation', 'inpainter',
                  'image_edit', 'custom_image_edit', 'manga_image_edit'),
    'tts': ('tts',),
}

_REQUEST_CONTEXT = ContextVar('api_key_request_context', default=None)
_ALIASES = dict.fromkeys(('imagegen', 'image_renderer', 'image_render', 'image_output'), 'image_generation')
_CORE_INHERITED_CONTEXTS = {
    'translation', 'refinement', 'glossary', 'glossary_refinement', 'summary', 'review',
    'metadata', 'book_title', 'batch_toc_translation', 'batch_header_translation',
    'qa_truncation', 'truncation',
}


def normalize_context(context):
    value = str(context or 'translation').strip().lower().replace(' ', '_').replace('-', '_')
    return _ALIASES.get(value, value)


def normalize_disabled_contexts(contexts):
    if not isinstance(contexts, (list, tuple, set)):
        return []
    return sorted({normalize_context(c) for c in contexts if isinstance(c, str) and c.strip()})


def current_key_context():
    return _REQUEST_CONTEXT.get() or 'translation'


def key_enabled_for_context(key, context=None):
    """Global disabling wins; absent route restrictions preserve legacy behavior."""
    if isinstance(key, dict):
        enabled = key.get('enabled', True)
        disabled = key.get('disabled_contexts', [])
    else:
        enabled = getattr(key, 'enabled', True)
        disabled = getattr(key, 'disabled_contexts', [])
    route = normalize_context(context if context is not None else current_key_context())
    blocked = normalize_disabled_contexts(disabled)
    return bool(enabled) and route not in blocked and not (route not in CONTEXT_LABELS and 'other' in blocked)


@contextmanager
def key_request_context(context):
    token = _REQUEST_CONTEXT.set(normalize_context(context))
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)


def route_key_context(method):
    """Bind routes across nested sends/retries without shared-client state races."""
    method_signature = signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        arguments = method_signature.bind(self, *args, **kwargs).arguments
        context = 'tts' if method.__name__ == 'text_to_speech' else arguments.get('context') or _REQUEST_CONTEXT.get()
        if not context:
            context = getattr(self, 'context', None)
            # Match _send_core's legacy inheritance rules. A previous image
            # request must not label the next ordinary text request as an image.
            if method.__name__ == '_send_core' and normalize_context(context) not in _CORE_INHERITED_CONTEXTS:
                context = None
        if not context and method.__name__ == '_send_core':
            thread_name = threading.current_thread().name
            if 'truncation' in thread_name.lower():
                context = 'truncation'
            elif 'Glossary' in thread_name:
                context = 'glossary'
        if not context:
            context = 'image_translation' if arguments.get('image_data') is not None else 'translation'
        with key_request_context(context):
            return method(self, *args, **kwargs)
    return wrapped
