"""
render_overlay_worker.py

Process-safe per-region overlay builder for manga rendering.
No imports from project modules to avoid side-effects in child processes.
"""
from typing import List, Tuple, Optional


def render_overlay_worker(spec: dict):
    try:
        from PIL import Image as _PIL, ImageDraw as _ImageDraw, ImageFont as _ImageFont

        idx: int = int(spec.get('idx', 0))
        cw, ch = spec.get('canvas_size', (0, 0))
        rx, ry, rw, rh = map(int, spec.get('rect', (0, 0, 0, 0)))
        txt: str = spec.get('text', '') or ''
        if spec.get('force_caps'):
            txt = txt.upper()
        font_path = spec.get('font_path')
        font_size_mode = spec.get('font_size_mode', 'multiplier')
        custom_font_size = int(spec.get('custom_font_size') or 0)
        text_color = tuple(spec.get('text_color', (0, 0, 0)))
        outline_color = tuple(spec.get('outline_color', (255, 255, 255)))
        ow_factor = max(1, int(spec.get('outline_width_factor', 24)))
        shadow_enabled = bool(spec.get('shadow_enabled', False))
        shadow_color = tuple(spec.get('shadow_color', (0, 0, 0)))
        shadow_dx, shadow_dy = tuple(spec.get('shadow_offset', (2, 2)))
        bg_style = str(spec.get('bg_style', 'circle'))
        bg_opacity = int(spec.get('bg_opacity', 130))
        bg_reduction = float(spec.get('bg_reduction', 1.0))
        draw_bg_only_when_free = bool(spec.get('draw_bg', True))

        # ROI overlay
        roi_w, roi_h = max(1, rw), max(1, rh)
        overlay = _PIL.new('RGBA', (roi_w, roi_h), (0, 0, 0, 0))
        draw = _ImageDraw.Draw(overlay)

        def _get_font(sz: int):
            try:
                if font_path:
                    return _ImageFont.truetype(font_path, sz)
            except Exception:
                pass
            try:
                return _ImageFont.load_default()
            except Exception:
                return _ImageFont.load_default()

        def _wrap(text: str, font, max_w: int):
            if not text:
                return ['']
            # Simple wrap by spaces/newlines
            blocks = text.replace('\r', '').split('\n')
            lines: List[str] = []
            for block in blocks:
                if not block:
                    lines.append('')
                    continue
                words = block.split(' ')
                cur = ''
                for w in words:
                    test = (cur + (' ' if cur else '') + w)
                    bbox = draw.textbbox((0, 0), test, font=font)
                    if (bbox[2] - bbox[0]) <= max_w or not cur:
                        cur = test
                    else:
                        lines.append(cur)
                        cur = w
                lines.append(cur)
            return lines if lines else ['']

        def _fit(text: str, max_w: int, max_h: int):
            if font_size_mode == 'fixed' and custom_font_size > 0:
                sz = custom_font_size
                f = _get_font(sz)
            else:
                sz = max(8, min(max_h, int(min(max_h, max_w) * 0.3)))
                f = _get_font(sz)
            lines = _wrap(text, f, max_w)
            bbox = draw.textbbox((0, 0), (lines[0] if lines else 'Ay'), font=f)
            lh = max(int(sz * 1.2), int((bbox[3] - bbox[1]) * 1.1))
            total_h = lh * max(1, len(lines))
            while (total_h > max_h or any((draw.textbbox((0, 0), ln, font=f)[2] - draw.textbbox((0, 0), ln, font=f)[0]) > max_w for ln in lines)) and sz > 8:
                sz -= 1
                f = _get_font(sz)
                lines = _wrap(text, f, max_w)
                bbox = draw.textbbox((0, 0), (lines[0] if lines else 'Ay'), font=f)
                lh = max(int(sz * 1.2), int((bbox[3] - bbox[1]) * 1.1))
                total_h = lh * max(1, len(lines))
            return sz, lines, lh

        font_size, lines, line_height = _fit(txt, roi_w, roi_h)
        font = _get_font(font_size)

        # Background
        if draw_bg_only_when_free and bg_opacity > 0:
            padding = int(font_size * 0.3)
            max_width = 0
            for ln in lines:
                bbox = draw.textbbox((0, 0), ln, font=font)
                max_width = max(max_width, bbox[2] - bbox[0])
            bg_w = int((max_width + padding * 2) * bg_reduction)
            bg_h = int(((len(lines) * line_height) + padding * 2) * bg_reduction)
            bg_x = (roi_w - bg_w) // 2
            bg_y = max(0, int((roi_h - bg_h) // 2))
            bg_rgba = (255, 255, 255, int(bg_opacity))
            if bg_style == 'box':
                r = max(4, min(20, bg_w // 10, bg_h // 10))
                draw.rectangle([bg_x + r, bg_y, bg_x + bg_w - r, bg_y + bg_h], fill=bg_rgba)
                draw.rectangle([bg_x, bg_y + r, bg_x + bg_w, bg_y + bg_h - r], fill=bg_rgba)
                draw.pieslice([bg_x, bg_y, bg_x + 2 * r, bg_y + 2 * r], 180, 270, fill=bg_rgba)
                draw.pieslice([bg_x + bg_w - 2 * r, bg_y, bg_x + bg_w, bg_y + 2 * r], 270, 360, fill=bg_rgba)
                draw.pieslice([bg_x, bg_y + bg_h - 2 * r, bg_x + 2 * r, bg_y + bg_h], 90, 180, fill=bg_rgba)
                draw.pieslice([bg_x + bg_w - 2 * r, bg_y + bg_h - 2 * r, bg_x + bg_w, bg_y + bg_h], 0, 90, fill=bg_rgba)
            elif bg_style == 'wrap':
                for i, ln in enumerate(lines):
                    bbox = draw.textbbox((0, 0), ln, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_bg_w = int((line_width + padding) * bg_reduction)
                    line_bg_x = (roi_w - line_bg_w) // 2
                    line_bg_y = int((roi_h - (len(lines) * line_height)) // 2 + i * line_height - padding // 2)
                    line_bg_h = int(line_height + padding // 2)
                    r = max(3, min(10, line_bg_w // 10, line_bg_h // 10))
                    draw.rectangle([line_bg_x + r, line_bg_y, line_bg_x + line_bg_w - r, line_bg_y + line_bg_h], fill=bg_rgba)
                    draw.rectangle([line_bg_x, line_bg_y + r, line_bg_x + line_bg_w, line_bg_y + line_bg_h - r], fill=bg_rgba)
                    draw.pieslice([line_bg_x, line_bg_y, line_bg_x + 2 * r, line_bg_y + 2 * r], 180, 270, fill=bg_rgba)
                    draw.pieslice([line_bg_x + line_bg_w - 2 * r, line_bg_y, line_bg_x + line_bg_w, line_bg_y + 2 * r], 270, 360, fill=bg_rgba)
                    draw.pieslice([line_bg_x, line_bg_y + line_bg_h - 2 * r, line_bg_x + 2 * r, line_bg_y + line_bg_h], 90, 180, fill=bg_rgba)
                    draw.pieslice([line_bg_x + line_bg_w - 2 * r, line_bg_y + line_bg_h - 2 * r, line_bg_x + line_bg_w, line_bg_y + line_bg_h], 0, 90, fill=bg_rgba)
            else:
                cx = roi_w // 2
                cy = roi_h // 2
                ew = int(bg_w * 1.2)
                eh = bg_h
                draw.ellipse([cx - ew // 2, cy - eh // 2, cx + ew // 2, cy + eh // 2], fill=bg_rgba)

        # Text
        total_h = line_height * max(1, len(lines))
        start_y = max(0, int((roi_h - total_h) // 2))
        ow = max(1, font_size // int(ow_factor))
        for i, ln in enumerate(lines):
            bbox = draw.textbbox((0, 0), ln, font=font)
            tw = bbox[2] - bbox[0]
            tx = max(0, int((roi_w - tw) // 2))
            ty = start_y + i * line_height
            if shadow_enabled:
                draw.text((tx + shadow_dx, ty + shadow_dy), ln, font=font, fill=shadow_color + (255,))
            if ow > 0:
                for dx in range(-ow, ow + 1):
                    for dy in range(-ow, ow + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((tx + dx, ty + dy), ln, font=font, fill=outline_color + (255,))
            draw.text((tx, ty), ln, font=font, fill=text_color + (255,))

        return (idx, rx, ry, roi_w, roi_h, overlay.tobytes())
    except Exception:
        return None
