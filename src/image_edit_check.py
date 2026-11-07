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
* ``redrawn_text_mask`` finds where a translate prompt (Translate SFX) redrew text, apart
  from the resampling / softening / local drift the model adds everywhere, so only that
  and the text mask are taken from the edit.
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
#: Translate prompts (``same_lettering``): the source lettering counts as still drawn only when the
#: edit's pixels there match the source's (normalised cross-correlation, best of the shifts within
#: ``LETTERING_SHIFT`` px, both softened by ``LETTERING_BLUR``) at least this well. Text the model left
#: in place keeps 0.9-1.0 even resampled and softened; text it redrew translated (new letters over the
#: old ones' place, whose ink overlaps the old strokes) scores about 0.3 (the owner's adventurer card), up
#: to about 0.55 for lines of Latin letters on the same baselines.
SAME_LETTERING_SIMILARITY = 0.75
LETTERING_SHIFT = 3
LETTERING_BLUR = 1.5


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

def lettering_similarity(source, edited, box, *, shift=LETTERING_SHIFT, blur=LETTERING_BLUR):
    """How well ``edited`` still shows the lettering ``source`` has in ``box`` (x, y, w, h): the best
    normalised cross-correlation of the two grey crops over shifts of up to ``shift`` px, both softened
    by a Gaussian of ``blur`` px (the model's resampling). 1.0 = the same letters; a redraw in other
    letters (a translation) or an emptied area scores low. None when the box is too small or flat."""
    cv2 = _cv2()
    x, y, w, h = (int(v) for v in box)
    height, width = source.shape[:2]
    x0, y0 = max(0, x - shift), max(0, y - shift)
    x1, y1 = min(width, x + w + shift), min(height, y + h + shift)
    if x1 - x0 <= 2 * shift + 4 or y1 - y0 <= 2 * shift + 4:
        return None
    src = cv2.GaussianBlur(_gray(source)[y0:y1, x0:x1].astype(np.float32), (0, 0), blur)
    out = cv2.GaussianBlur(_gray(edited)[y0:y1, x0:x1].astype(np.float32), (0, 0), blur)
    template = src[shift:-shift, shift:-shift]
    if float(template.std()) < 1.0:
        return None
    if float(out.std()) < 1.0:
        return 0.0  # emptied
    return float(cv2.matchTemplate(out, template, cv2.TM_CCOEFF_NORMED).max())


def unremoved_text_mask(source, edited, text_mask, *, kept_threshold=KEPT_INK_THRESHOLD,
                        min_stroke_pixels=MIN_STROKE_PIXELS, same_lettering=False):
    """Return ``(failed_mask, stats)`` for the text-mask components whose ink is still there.

    ``failed_mask`` is a uint8 mask (255 = component whose text the edit left). ``stats`` lists
    one dict per judged component: ``bbox`` (x, y, w, h), ``strokes``, ``kept`` (share of
    source strokes still drawn) and ``failed``.

    ``same_lettering`` (a translate image edit prompt, which asks the model to redraw some text in
    the target language): ink in the component's place is not enough, the source's own letters must
    still be drawn there (``lettering_similarity`` at least ``SAME_LETTERING_SIMILARITY``, added to
    the stats as ``similarity``). A card whose Korean the model replaced with English keeps most of
    its ink where the old strokes were, but not the same letters, so it passes.
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
        entry = {'bbox': (x, y, bw, bh), 'strokes': stroke_count, 'kept': share}
        if is_failed and same_lettering:
            ys, xs = np.nonzero(judged)
            inner_box = (x + int(xs.min()), y + int(ys.min()),
                         int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)
            similarity = lettering_similarity(source, edited, inner_box)
            entry['similarity'] = similarity
            is_failed = similarity is None or similarity >= SAME_LETTERING_SIMILARITY
        if is_failed:
            failed[y:y + bh, x:x + bw][component] = 255
        entry['failed'] = is_failed
        stats.append(entry)
    return failed, stats


def edit_footprint(source, edited, failed_mask, text_mask):
    """Return ``(footprint, redo_text)`` around the failed text components.

    ``footprint`` (bool) is every connected area the edit changed that touches a failed
    component, near that component; putting ``source`` back there undoes the edit for that
    text only. ``redo_text`` (uint8, 255) is the part of ``text_mask`` inside it: the text
    that then needs cleaning again.

    A model that returns the whole crop re-encoded changes every pixel of it, so the area is
    kept within half the failed component's size around it and out of the text components
    the edit did clean (their edit stays).
    """
    cv2 = _cv2()
    failed = _binary(failed_mask)
    mask = _binary(text_mask)
    if source.ndim == 3:
        changed = np.any(source != edited, axis=2)
    else:
        changed = source != edited
    height, width = failed.shape[:2]
    reach = np.zeros((height, width), bool)
    count, _labels, boxes, _ = cv2.connectedComponentsWithStats(failed, connectivity=8)
    for label in range(1, count):
        x, y, bw, bh = (int(v) for v in boxes[label][:4])
        pad = max(8, int(0.5 * max(bw, bh)))
        reach[max(0, y - pad):min(height, y + bh + pad), max(0, x - pad):min(width, x + bw + pad)] = True
    passed = ((mask > 0) & (failed == 0)).astype(np.uint8)
    if passed.any():
        passed = cv2.dilate(passed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    changed = (changed & reach & (passed == 0)) | (failed > 0)
    changed = cv2.dilate(changed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    changed[(passed > 0) & (failed == 0)] = 0
    count, labels = cv2.connectedComponents(changed, connectivity=8)
    hit = np.unique(labels[failed > 0])
    hit = hit[hit > 0]
    footprint = np.isin(labels, hit) if hit.size else np.zeros(failed.shape, bool)
    redo = ((mask > 0) & footprint).astype(np.uint8) * 255
    return footprint, redo


# ---------------------------------------------------------------------------
# text a translate prompt redrew
# ---------------------------------------------------------------------------

#: Colour levels (any channel) between source and edit for a pixel to count as changed.
REDRAW_DIFF = 40
#: Opening kernel (px): drops the 1-2 px edge differences a resampled / softened copy has
#: along every line, keeps the strokes of letters (small sign text included).
REDRAW_OPEN = 3
#: Closing kernel (px): joins the strokes of one word / sound effect into a single area.
REDRAW_CLOSE = 11
#: Smallest area (px, after closing) and side (px) of a redrawn piece of text.
REDRAW_MIN_AREA = 200
REDRAW_MIN_SIDE = 12
#: Share of an area's pixels that must be changed strokes (not just closed-over gaps).
REDRAW_MIN_DENSITY = 0.15
#: Local alignment search: blocks of this size (px) whose changes a shift of the edit by
#: up to ``REDRAW_SHIFT_REACH`` px explains (best shift leaves at most
#: ``REDRAW_SHIFT_EXPLAINED`` of the changed pixels) are art the model moved or stretched,
#: not text it redrew.
REDRAW_SHIFT_BLOCK = 32
REDRAW_SHIFT_REACH = 40
REDRAW_SHIFT_EXPLAINED = 0.3
#: A kept area takes in the changed strokes joined to it only while they stay within this many
#: times its own area.
REDRAW_JOIN_GROWTH = 2.0
#: A kept area larger than this share of the crop is not a word but art the model stretched or
#: restyled (a whole floor redrawn lower); it is dropped.
REDRAW_MAX_SHARE = 0.12
#: Margin (px) added around each redrawn area: the caller's feather (Gaussian, sigma 5) then fades the
#: edit out outside the letters, not across their edges.
REDRAW_GROW = 12


def _shift_explained_blocks(source, edited, changed, *, block=REDRAW_SHIFT_BLOCK, reach=REDRAW_SHIFT_REACH,
                            explained=REDRAW_SHIFT_EXPLAINED, threshold=REDRAW_DIFF):
    """Bool mask of the blocks whose changed pixels a local shift of ``edited`` explains.

    Image-edit models return the crop resampled and often locally moved or stretched (a figure
    a few pixels lower, a face slightly larger). Those areas differ a lot pixel by pixel but
    match after a shift; redrawn text matches under no shift. Searched on a quarter-size copy.
    """
    cv2 = _cv2()
    height, width = changed.shape[:2]
    factor = 4 if min(height, width) >= 4 * block else 1
    small_w, small_h = max(1, width // factor), max(1, height // factor)
    src = cv2.resize(source, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.int16)
    out = cv2.resize(edited, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.int16)
    if src.ndim == 2:
        src, out = src[:, :, None], out[:, :, None]
    cell = max(1, block // factor)
    radius = max(1, reach // factor)
    rows, cols = -(-small_h // cell), -(-small_w // cell)
    pad_h, pad_w = rows * cell - small_h, cols * cell - small_w

    def per_block(diff):
        counts = np.pad((np.max(diff, axis=2) > threshold).astype(np.float32), ((0, pad_h), (0, pad_w)))
        return counts.reshape(rows, cell, cols, cell).sum(axis=(1, 3))

    base = per_block(np.abs(src - out))
    best = base.copy()
    out_padded = np.pad(out, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
    src_padded = np.pad(src, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx or dy:
                # Both ways: the edit shifted matches the source, and the source shifted back matches
                # the edit. New lettering on a flat background passes only the first (a shift brings
                # in plain background); the edit's letters appear nowhere in the source.
                out_shifted = out_padded[radius + dy:radius + dy + small_h, radius + dx:radius + dx + small_w]
                src_shifted = src_padded[radius - dy:radius - dy + small_h, radius - dx:radius - dx + small_w]
                both = np.maximum(per_block(np.abs(src - out_shifted)), per_block(np.abs(out - src_shifted)))
                np.minimum(best, both, out=best)
    blocks = (base > 0) & (best <= explained * base)
    grid = np.repeat(np.repeat(blocks, cell, axis=0), cell, axis=1)[:small_h, :small_w].astype(np.uint8)
    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST) > 0


def redrawn_text_mask(source, edited):
    """Return ``(mask, stats)``: where ``edited`` redrew text, as opposed to drifted art.

    A translate image edit prompt (Translate SFX) asks the model to redraw sound effects and
    background text translated, but the model returns the whole crop resampled, softened and
    partly moved. This keeps the areas that clearly changed like text: pixels that differ by
    more than ``REDRAW_DIFF``, without the thin edge differences (opening), outside blocks a
    local shift explains, joined per word (closing), large and dense enough. Each kept area then
    takes in the changed strokes joined to it within half its size around it, also those a shift
    explained (a redrawn word in the same style as the source word can match it in places), unless
    that more than ``REDRAW_JOIN_GROWTH`` times its area (moved art around it); an area larger
    than ``REDRAW_MAX_SHARE`` of the crop is dropped (stretched art). The rest is
    filled to its convex hull (the whole word, its letters' gaps included) and grown by
    ``REDRAW_GROW`` px. ``mask`` is uint8 (255 = keep the edit); ``stats`` lists the kept
    areas' boxes.
    """
    cv2 = _cv2()
    stats = []
    if source is None or edited is None or source.shape != edited.shape:
        return None, stats
    height, width = source.shape[:2]
    keep = np.zeros((height, width), np.uint8)
    if height < REDRAW_MIN_SIDE or width < REDRAW_MIN_SIDE:
        return keep, stats
    diff = np.abs(source.astype(np.int16) - edited.astype(np.int16))
    if diff.ndim == 3:
        diff = np.max(diff, axis=2)
    changed = (diff > REDRAW_DIFF).astype(np.uint8)

    def kernel(size):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    strokes = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel(REDRAW_OPEN))
    if not strokes.any():
        return keep, stats
    all_joined = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel(REDRAW_CLOSE))
    _all_count, all_labels = cv2.connectedComponents(all_joined, connectivity=8)
    strokes[_shift_explained_blocks(source, edited, strokes)] = 0
    joined = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, kernel(REDRAW_CLOSE))
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    for label in range(1, count):
        x, y, bw, bh, area = (int(v) for v in boxes[label])
        if area < REDRAW_MIN_AREA or min(bw, bh) < REDRAW_MIN_SIDE:
            continue
        component = labels[y:y + bh, x:x + bw] == label
        if float(np.mean(strokes[y:y + bh, x:x + bw][component])) < REDRAW_MIN_DENSITY:
            continue
        pad_x, pad_y = bw // 2, bh // 2
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + bw + pad_x), min(height, y + bh + pad_y)
        near = all_labels[y0:y1, x0:x1]
        joined_to = np.unique(all_labels[y:y + bh, x:x + bw][component])
        area_mask = np.isin(near, joined_to[joined_to > 0]).astype(np.uint8)
        if np.count_nonzero(area_mask) > REDRAW_JOIN_GROWTH * area:
            area_mask[:] = 0  # joined to a large area (art the model moved): the word alone
        area_mask[y - y0:y - y0 + bh, x - x0:x - x0 + bw][component] = 1
        hull = None
        for candidate in (area_mask, None):
            if candidate is None:  # the grown area is too large: the word alone
                candidate = np.zeros_like(area_mask)
                candidate[y - y0:y - y0 + bh, x - x0:x - x0 + bw][component] = 1
            points = cv2.findNonZero(candidate)
            hull = cv2.convexHull(points + np.array([x0, y0], dtype=points.dtype))
            if cv2.contourArea(hull) <= REDRAW_MAX_SHARE * height * width:
                break
            hull = None
        if hull is None:
            continue  # a large part of the crop moved or restyled (stretched art), not a word
        cv2.fillConvexPoly(keep, hull, 255)
        stats.append({'bbox': (x, y, bw, bh), 'area': area})
    if stats:
        keep = cv2.dilate(keep, kernel(2 * REDRAW_GROW + 1))
    return keep, stats


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


#: Marks that are letters to Unicode but read as a line: the katakana long-vowel mark and
#: wave dashes (a dashed bubble border often reads as one of these).
_DASH_LETTERS = frozenset('ー〜～ｰ')


def is_symbol_only_text(text):
    """True for text with no letters or digits: punctuation and symbols only (…, ?!, ♡, —, ✉)."""
    compact = _compact(text)
    if not compact:
        return False
    return all(ch in _DASH_LETTERS or unicodedata.category(ch)[0] in ('P', 'S') for ch in compact)


def number_text_unchanged(source_text, translated_text):
    """True when both texts are number-only and the same after NFKC and without spaces."""
    return (is_number_only_text(source_text) and is_number_only_text(translated_text)
            and _compact(source_text) == _compact(translated_text))


def region_needs_no_lettering(source_text, translated_text):
    """A region with nothing to translate keeps its art when its translation adds nothing.

    Number-only text (11:59) keeps it while the translation is missing or the same digits.
    Symbol-only text (…, ?!, ♡, a dash read off a dashed border, an icon) keeps it while the
    translation is missing or symbols only as well: the drawn marks already say the same.
    """
    if is_number_only_text(source_text):
        if not str(translated_text or '').strip():
            return True
        return number_text_unchanged(source_text, translated_text)
    if is_symbol_only_text(source_text):
        return not str(translated_text or '').strip() or is_symbol_only_text(translated_text)
    return False
