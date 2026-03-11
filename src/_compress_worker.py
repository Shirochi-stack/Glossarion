# _compress_worker.py - Lightweight worker for image compression
# This module is intentionally minimal so ProcessPoolExecutor workers
# don't reimport the heavy epub_converter.py (21s of imports).
# Only stdlib 'os' is imported at module level; PIL is imported inside the function.

import os


def _compress_single_image(images_dir, original_name, safe_name, quality, is_gif):
    """Module-level worker for compressing a single image (ProcessPoolExecutor-compatible).
    
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
