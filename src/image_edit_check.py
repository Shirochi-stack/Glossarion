"""Checks for text-removal output (Custom image edit) and number-only regions.

Image-edit models (Nano Banana and the like) return the crop at their own size and
sometimes move or rescale the content inside it, and sometimes return the text untouched.
This module holds the measurements the manga pipeline uses on that output:

* ``register_to_source`` aligns a returned crop to the source crop. The transform
  (translation + scale, small rotation/shear allowed) is estimated with OpenCV's ECC on the
  pixels outside the text mask, the only pixels the model was asked to keep. It is applied
  only when it makes those pixels match better.
* ``unremoved_text_mask`` finds the text-mask components whose source lettering is still
  in the edited image: the source's ink strokes (pixels far from their local background)
  are looked up in the edited image; when most of them are still there, the text was not
  removed.
* ``edit_footprint`` gives the area the edit changed around such components, so the
  caller can put the source art back there (or hand it to another inpainter once).
* ``is_number_only_text`` / ``number_text_unchanged`` describe regions such as clocks
  (11:59) whose translation is the same digits: they need no cleaning and no lettering.

Pure numpy/OpenCV, no GUI and no network. Python 3.10 compatible.
"""

import unicodedata

import numpy as np

#: A text component counts as "not removed" when at least this share of its source ink
#: strokes is still drawn in the edited image (removed text keeps about 0-25%: the bubble
#: outline caught by the mask margin; text the model left in place keeps 60-100%).
KEPT_INK_THRESHOLD = 0.5
#: Components with fewer source stroke pixels than this are not judged (nothing to read).
MIN_STROKE_PIXELS = 12
#: Share of a component's width/height left out on each side when judging it.
INNER_MARGIN = 0.15
#: Grey levels between a stroke pixel and its local background.
INK_CONTRAST = 48
#: Window of the local-background median (wider than a glyph stroke).
BACKGROUND_WINDOW = 21


def _cv2():
    import cv2
    return cv2


def _gray(image):
    cv2 = _cv2()
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _binary(mask):
    cv2 = _cv2()
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def _ink(gray):
    """(dark strokes, light strokes) of a grey image against its local background."""
    cv2 = _cv2()
    background = cv2.medianBlur(gray, BACKGROUND_WINDOW).astype(np.int16)
    signed = gray.astype(np.int16) - background
    return signed <= -INK_CONTRAST, signed >= INK_CONTRAST


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def register_to_source(source, edited, text_mask, *, max_shift_fraction=0.08, max_scale_change=0.15):
    """Align ``edited`` (same size as ``source``) to ``source`` outside ``text_mask``.

    Returns ``(aligned, info)``. ``info['applied']`` says whether a transform was used;
    ``dx``/``dy`` (pixels) and ``scale_x``/``scale_y`` describe it; ``error_before`` and
    ``error_after`` are the mean absolute grey difference outside the mask. ``edited`` is
    returned unchanged when the estimate fails, is implausible or does not help.
    """
    cv2 = _cv2()
    info = {'applied': False, 'dx': 0.0, 'dy': 0.0, 'scale_x': 1.0, 'scale_y': 1.0,
            'error_before': 0.0, 'error_after': 0.0}
    if source is None or edited is None or source.shape != edited.shape:
        return edited, info
    h, w = source.shape[:2]
    if h < 32 or w < 32:
        return edited, info

    mask = _binary(text_mask) if text_mask is not None else np.zeros((h, w), np.uint8)
    margin = max(5, int(round(0.01 * (h * h + w * w) ** 0.5)))
    keep_out = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    valid = keep_out == 0
    if np.count_nonzero(valid) < 0.1 * h * w:
        return edited, info

    src_gray = _gray(source)
    out_gray = _gray(edited)
    before = float(np.mean(np.abs(src_gray.astype(np.int16) - out_gray.astype(np.int16))[valid]))
    info['error_before'] = info['error_after'] = before
    if before < 1.0:
        return edited, info  # already matches outside the text

    # ECC on a reduced copy (it is iterative); the translation is scaled back up after.
    factor = min(1.0, 1024.0 / float(max(h, w)))
    if factor < 1.0:
        size = (max(16, int(round(w * factor))), max(16, int(round(h * factor))))
        src_small = cv2.resize(src_gray, size, interpolation=cv2.INTER_AREA)
        out_small = cv2.resize(out_gray, size, interpolation=cv2.INTER_AREA)
        valid_small = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    else:
        src_small, out_small, valid_small = src_gray, out_gray, valid.astype(np.uint8)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            src_small.astype(np.float32), out_small.astype(np.float32), warp,
            cv2.MOTION_AFFINE, criteria, valid_small, 5,
        )
    except cv2.error:
        return edited, info
    if factor < 1.0:
        warp = warp.copy()
        warp[0, 2] /= factor
        warp[1, 2] /= factor

    linear = warp[:, :2]
    scale_x = float(np.hypot(linear[0, 0], linear[1, 0]))
    scale_y = float(np.hypot(linear[0, 1], linear[1, 1]))
    shear = abs(float(linear[0, 1] + linear[1, 0]))
    dx, dy = float(warp[0, 2]), float(warp[1, 2])
    if (abs(scale_x - 1.0) > max_scale_change or abs(scale_y - 1.0) > max_scale_change
            or shear > 0.05 or abs(dx) > max_shift_fraction * w + 2 or abs(dy) > max_shift_fraction * h + 2):
        return edited, info

    # warp maps source coordinates into the edited image: sample edited there.
    aligned = cv2.warpAffine(edited, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    after = float(np.mean(np.abs(src_gray.astype(np.int16) - _gray(aligned).astype(np.int16))[valid]))
    if after >= before * 0.9:
        return edited, info
    info.update({'applied': True, 'dx': dx, 'dy': dy, 'scale_x': scale_x, 'scale_y': scale_y,
                 'error_after': after})
    return aligned, info


# ---------------------------------------------------------------------------
# text left in place
# ---------------------------------------------------------------------------

def unremoved_text_mask(source, edited, text_mask, *, kept_threshold=KEPT_INK_THRESHOLD,
                        min_stroke_pixels=MIN_STROKE_PIXELS):
    """Return ``(failed_mask, stats)`` for the text-mask components whose ink is still there.

    ``failed_mask`` is a uint8 mask (255 = component whose text the edit left). ``stats`` lists
    one dict per judged component: ``bbox`` (x, y, w, h), ``strokes``, ``kept`` (share of
    source strokes still drawn) and ``failed``.
    """
    cv2 = _cv2()
    mask = _binary(text_mask)
    failed = np.zeros(mask.shape, np.uint8)
    stats = []
    if source is None or edited is None or source.shape[:2] != edited.shape[:2] or not mask.any():
        return failed, stats

    src_dark, src_light = _ink(_gray(source))
    out_dark, out_light = _ink(_gray(edited))
    near = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 2 px tolerance
    out_dark = cv2.dilate(out_dark.astype(np.uint8), near) > 0
    out_light = cv2.dilate(out_light.astype(np.uint8), near) > 0
    strokes = (src_dark | src_light) & (mask > 0)
    kept_px = (src_dark & out_dark) | (src_light & out_light)

    count, labels, boxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, count):
        x, y, bw, bh, _area = (int(v) for v in boxes[label])
        component = labels[y:y + bh, x:x + bw] == label
        # Judge the inner part: the mask margin holds bubble outlines and art the edit
        # rightly keeps; the lettering sits in the middle.
        ex, ey = max(2, int(bw * INNER_MARGIN)), max(2, int(bh * INNER_MARGIN))
        inner = cv2.erode(component.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ex + 1, 2 * ey + 1)),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        judged = inner if inner.any() else component
        stroke_count = int(np.count_nonzero(strokes[y:y + bh, x:x + bw] & judged))
        if stroke_count < min_stroke_pixels:
            continue
        kept = int(np.count_nonzero(kept_px[y:y + bh, x:x + bw] & strokes[y:y + bh, x:x + bw] & judged))
        share = kept / float(stroke_count)
        is_failed = share >= kept_threshold
        if is_failed:
            failed[y:y + bh, x:x + bw][component] = 255
        stats.append({'bbox': (x, y, bw, bh), 'strokes': stroke_count, 'kept': share, 'failed': is_failed})
    return failed, stats


def edit_footprint(source, edited, failed_mask, text_mask):
    """Return ``(footprint, redo_text)`` around the failed text components.

    ``footprint`` (bool) is every connected area the edit changed that touches a failed
    component; putting ``source`` back there undoes the edit for that text only.
    ``redo_text`` (uint8, 255) is the part of ``text_mask`` inside it: the text that then
    needs cleaning again, including passing components the same area swallowed.
    """
    cv2 = _cv2()
    failed = _binary(failed_mask)
    mask = _binary(text_mask)
    if source.ndim == 3:
        changed = np.any(source != edited, axis=2)
    else:
        changed = source != edited
    changed = (changed | (failed > 0)).astype(np.uint8)
    changed = cv2.dilate(changed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    count, labels = cv2.connectedComponents(changed, connectivity=8)
    hit = np.unique(labels[failed > 0])
    hit = hit[hit > 0]
    footprint = np.isin(labels, hit) if hit.size else np.zeros(failed.shape, bool)
    redo = ((mask > 0) & footprint).astype(np.uint8) * 255
    return footprint, redo


# ---------------------------------------------------------------------------
# number-only regions
# ---------------------------------------------------------------------------

def _compact(text):
    return ''.join(unicodedata.normalize('NFKC', str(text or '')).split())


def is_number_only_text(text):
    """True for text made only of digits, punctuation and symbols (11:59, 12:00~13:00, 3/4)."""
    compact = _compact(text)
    if not compact or not any(ch.isdigit() for ch in compact):
        return False
    for ch in compact:
        if ch.isdigit():
            continue
        category = unicodedata.category(ch)
        if category[0] in ('P', 'S'):
            continue
        return False
    return True


def number_text_unchanged(source_text, translated_text):
    """True when both texts are number-only and the same after NFKC and without spaces."""
    return (is_number_only_text(source_text) and is_number_only_text(translated_text)
            and _compact(source_text) == _compact(translated_text))


def region_needs_no_lettering(source_text, translated_text):
    """A number-only region keeps its art when its translation is missing or unchanged."""
    if not is_number_only_text(source_text):
        return False
    if not str(translated_text or '').strip():
        return True
    return number_text_unchanged(source_text, translated_text)
