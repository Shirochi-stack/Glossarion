#!/usr/bin/env python3
"""Lightweight standalone image compression worker.

This module is intentionally minimal so that when spawned as a subprocess
via `python _compress_worker.py`, it avoids importing the heavy GUI/EPUB
modules (ebooklib, bs4, translator_gui, etc.) that take ~22s on Windows.

When run as __main__, it reads JSON work items from stdin (one per line)
and writes JSON results to stdout (one per line). A blank/empty line or
EOF signals shutdown.
"""

import os
import sys
import json


def _compress_single_image(images_dir, original_name, safe_name, quality, is_gif):
    """Compress a single image (ProcessPoolExecutor / subprocess compatible).

    Returns dict with: original_name, safe_name, new_safe_name, status,
                       original_size, compressed_size, error
    """
    try:
        from PIL import Image
        img_path = os.path.join(images_dir, original_name)

        if not os.path.isfile(img_path):
            return {
                'original_name': original_name, 'safe_name': safe_name,
                'new_safe_name': safe_name, 'status': 'missing',
                'original_size': 0, 'compressed_size': 0, 'error': None
            }

        original_size = os.path.getsize(img_path)

        if is_gif:
            # Compress GIF in place
            im = Image.open(img_path)
            if hasattr(im, 'n_frames') and im.n_frames > 1:
                frames = []
                try:
                    while True:
                        frames.append(im.copy())
                        im.seek(im.tell() + 1)
                except EOFError:
                    pass
                if frames:
                    frames[0].save(
                        img_path, save_all=True, append_images=frames[1:],
                        optimize=True, loop=im.info.get('loop', 0)
                    )
            else:
                im.save(img_path, optimize=True)
            im.close()
            compressed_size = os.path.getsize(img_path)
            return {
                'original_name': original_name, 'safe_name': safe_name,
                'new_safe_name': safe_name, 'status': 'compressed',
                'original_size': original_size, 'compressed_size': compressed_size, 'error': None
            }
        else:
            # Convert to .webp
            webp_name = os.path.splitext(safe_name)[0] + '.webp'
            webp_path = os.path.join(images_dir, os.path.splitext(original_name)[0] + '.webp')

            im = Image.open(img_path)
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                im = im.convert('RGBA')
            elif im.mode != 'RGB':
                im = im.convert('RGB')

            im.save(webp_path, 'WEBP', quality=quality, method=4)
            im.close()

            # Remove original if webp was created successfully and is different file
            if os.path.exists(webp_path) and webp_path != img_path:
                try:
                    os.remove(img_path)
                except Exception:
                    pass

            compressed_size = os.path.getsize(webp_path) if os.path.exists(webp_path) else 0
            return {
                'original_name': original_name, 'safe_name': safe_name,
                'new_safe_name': webp_name, 'status': 'compressed',
                'original_size': original_size, 'compressed_size': compressed_size, 'error': None
            }
    except Exception as e:
        return {
            'original_name': original_name, 'safe_name': safe_name,
            'new_safe_name': safe_name, 'status': 'failed',
            'original_size': 0, 'compressed_size': 0, 'error': str(e)
        }


# ── Subprocess worker entry point ──────────────────────────────────────────
if __name__ == '__main__':
    # Reconfigure stdout for UTF-8 on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stdin, 'reconfigure'):
        try:
            sys.stdin.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    # Pre-import PIL so the first task doesn't pay the import cost
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        pass

    # Signal readiness
    sys.stdout.write('READY\n')
    sys.stdout.flush()

    # Process work items from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        try:
            job = json.loads(line)
            result = _compress_single_image(
                job['images_dir'], job['original_name'],
                job['safe_name'], job['quality'], job['is_gif']
            )
            sys.stdout.write(json.dumps(result) + '\n')
            sys.stdout.flush()
        except Exception as e:
            err_result = {
                'original_name': job.get('original_name', '?'),
                'safe_name': job.get('safe_name', '?'),
                'new_safe_name': job.get('safe_name', '?'),
                'status': 'failed',
                'original_size': 0, 'compressed_size': 0,
                'error': str(e)
            }
            sys.stdout.write(json.dumps(err_result) + '\n')
            sys.stdout.flush()
