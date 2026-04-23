"""
PixelHacker inference helper.

Wraps the upstream hustvl/PixelHacker repo (Apache-2.0) so Glossarion's
LocalInpainter can use it without forcing users to manually clone a second
project. The model architecture and pipeline are complex (~1000 lines across
several files) so we do NOT re-vendor them here. Instead, on first use we
fetch the required files from GitHub into `src/pixelhacker_upstream/` and
import from there.

Upstream commit: 25eefb4fc6d7af7cf59f046f6c412cb259e17491
Upstream repo:   https://github.com/hustvl/PixelHacker
"""
from __future__ import annotations

import os
import sys
import logging
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Files pulled from upstream on first use. Paths are relative to the repo root
# and are mirrored into `_UPSTREAM_DIR` with the same layout.
_UPSTREAM_COMMIT = "25eefb4fc6d7af7cf59f046f6c412cb259e17491"
_UPSTREAM_BASE = (
    f"https://raw.githubusercontent.com/hustvl/PixelHacker/{_UPSTREAM_COMMIT}"
)
_UPSTREAM_FILES = [
    "gla_model/__init__.py",
    "gla_model/gla.py",
    "gla_model/PixelHacker.py",
    "pipeline.py",
    "utils.py",
    "config/PixelHacker_sdvae_f8d4.yaml",
    "vae/config.json",
]

_UPSTREAM_DIR = Path(__file__).resolve().parent / "pixelhacker_upstream"


class PixelHackerUnavailableError(RuntimeError):
    """Raised when PixelHacker cannot be set up (e.g. offline, missing deps)."""


def _check_diffusers() -> None:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise PixelHackerUnavailableError(
            "PixelHacker requires the 'diffusers' and 'torch' libraries. "
            "Install with:\n    pip install 'diffusers>=0.30.2' 'transformers>=4.40' accelerate"
        ) from e


def _download_upstream_file(rel_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_UPSTREAM_BASE}/{rel_path}"
    logger.info(f"[PixelHacker] Fetching {rel_path} from upstream...")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
    except Exception as e:
        raise PixelHackerUnavailableError(
            f"Failed to download upstream PixelHacker file '{rel_path}'. "
            f"Check your internet connection or manually clone "
            f"https://github.com/hustvl/PixelHacker into {_UPSTREAM_DIR}.\n"
            f"Underlying error: {e}"
        ) from e


def ensure_pixelhacker_available() -> Path:
    """Make sure the vendored upstream code is present and importable.

    Returns the path to the upstream directory and adds it to sys.path.
    Raises PixelHackerUnavailableError on failure.
    """
    _check_diffusers()
    _UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure each required file is present. We intentionally don't checksum
    # these; if a user has a modified copy locally we respect it.
    missing = []
    for rel in _UPSTREAM_FILES:
        dest = _UPSTREAM_DIR / rel
        if not dest.exists() or dest.stat().st_size == 0:
            missing.append((rel, dest))

    if missing:
        logger.info(
            f"[PixelHacker] Setting up upstream code ({len(missing)} files)..."
        )
        for rel, dest in missing:
            _download_upstream_file(rel, dest)

    # Make the module importable.
    upstream_str = str(_UPSTREAM_DIR)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)
    return _UPSTREAM_DIR


class PixelHackerRunner:
    """High-level wrapper around the upstream PixelHacker_Pipeline.

    Exposes a single `inpaint(image_rgb_uint8, mask_uint8)` method that does
    the necessary preprocessing, runs the pipeline, and returns a uint8 RGB
    image of the same size as the input.
    """

    def __init__(
        self,
        model_weight_path: str,
        vae_weight_path: str,
        device: str = "cuda",
        dtype: str = "fp16",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
    ) -> None:
        ensure_pixelhacker_available()

        import torch
        from diffusers import DDIMScheduler, AutoencoderKL  # noqa: F401

        # Import from the now-available upstream module.
        # `utils.py` provides build_model + build_vae + load_cfg.
        import utils as _ph_utils  # type: ignore
        from utils import load_cfg, build_model, build_vae  # type: ignore
        from pipeline import PixelHacker_Pipeline  # type: ignore

        # The upstream `load_cfg` only needs `omegaconf` when it's handed a
        # Python dict (it wraps it in an OmegaConf DictConfig). All downstream
        # consumers access it with plain dict-style keys (`cfg['data']['image_size']`
        # etc.), so a plain dict works just as well. Patch the function to
        # drop the omegaconf dependency entirely.
        _orig_load_cfg = _ph_utils.load_cfg

        def _patched_load_cfg(cfg):
            if isinstance(cfg, dict):
                return cfg
            return _orig_load_cfg(cfg)

        _ph_utils.load_cfg = _patched_load_cfg

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(dtype.lower(), torch.float32)

        # Resolve device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("[PixelHacker] CUDA requested but unavailable; using CPU")
            device = "cpu"
        self.device = device
        self.dtype = torch_dtype
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)

        cfg_path = _UPSTREAM_DIR / "config" / "PixelHacker_sdvae_f8d4.yaml"
        model_cfg = load_cfg(str(cfg_path))

        # The upstream model expects its VAE directory layout. We stage the
        # downloaded files into a pair of tmp directories that match.
        # build_vae reads model_cfg['vae']['model_dir'] relative to cwd, so we
        # override it to the absolute path of our staging dir.
        vae_stage = self._stage_vae_dir(vae_weight_path)
        model_cfg = dict(model_cfg)  # shallow copy
        model_cfg["vae"] = dict(model_cfg.get("vae", {}))
        model_cfg["vae"]["model_dir"] = str(vae_stage)

        logger.info("[PixelHacker] Building model...")
        model = build_model(model_cfg, 20).to(self.device)
        state = torch.load(model_weight_path, map_location=self.device)
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if missing_keys:
            logger.warning(
                f"[PixelHacker] {len(missing_keys)} missing keys when "
                f"loading weights (sample: {missing_keys[:3]})"
            )
        if unexpected_keys:
            logger.warning(
                f"[PixelHacker] {len(unexpected_keys)} unexpected keys "
                f"(sample: {unexpected_keys[:3]})"
            )
        model.eval()
        if self.device != "cpu":
            try:
                model = model.to(dtype=torch_dtype)
            except Exception:
                logger.info("[PixelHacker] Could not cast model dtype; staying fp32")
                self.dtype = torch.float32

        logger.info("[PixelHacker] Building VAE...")
        vae = build_vae(model_cfg).to(self.device)
        if self.device != "cpu":
            try:
                vae = vae.to(dtype=self.dtype)
            except Exception:
                pass

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )

        self._pipe = PixelHacker_Pipeline(
            model=model,
            vae=vae,
            scheduler=scheduler,
            device=self.device,
            dtype=self.dtype,
        )

        vae_ds = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_size = int(model.diff_model.config.sample_size * vae_ds)
        # Sanity: the upstream assert is image_size == model_cfg['data']['image_size']
        if self.image_size != int(model_cfg["data"]["image_size"]):
            logger.warning(
                f"[PixelHacker] Image size mismatch (derived {self.image_size} "
                f"vs cfg {model_cfg['data']['image_size']}); using derived value"
            )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _stage_vae_dir(self, vae_weight_path: str) -> Path:
        """Produce a directory laid out as {dir}/config.json and
        {dir}/diffusion_pytorch_model.bin so `build_vae` can load it."""
        stage = _UPSTREAM_DIR / "vae_staged"
        stage.mkdir(parents=True, exist_ok=True)
        # Copy config.json if not already there.
        cfg_src = _UPSTREAM_DIR / "vae" / "config.json"
        cfg_dst = stage / "config.json"
        if not cfg_dst.exists() or cfg_dst.stat().st_size == 0:
            cfg_dst.write_bytes(cfg_src.read_bytes())
        # Symlink or copy the weight file.
        weight_dst = stage / "diffusion_pytorch_model.bin"
        if weight_dst.exists():
            # If it already points at the right file, trust it.
            try:
                if os.path.samefile(weight_dst, vae_weight_path):
                    return stage
            except Exception:
                pass
            try:
                weight_dst.unlink()
            except Exception:
                pass
        try:
            # Prefer a symlink; fall back to copy on Windows without permission.
            os.symlink(vae_weight_path, weight_dst)
        except Exception:
            import shutil
            shutil.copy2(vae_weight_path, weight_dst)
        return stage

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def inpaint(self, image_rgb_uint8, mask_uint8):
        """Run PixelHacker inpainting.

        Args:
            image_rgb_uint8: HxWx3 uint8 RGB image.
            mask_uint8: HxW uint8 mask (255 = inpaint, 0 = keep).
        Returns:
            HxWx3 uint8 RGB image (same size as input).
        """
        from PIL import Image
        import numpy as np

        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
            raise ValueError("image must be HxWx3 RGB")
        if mask_uint8.ndim != 2:
            raise ValueError("mask must be HxW grayscale")

        orig_h, orig_w = image_rgb_uint8.shape[:2]

        img_pil = Image.fromarray(image_rgb_uint8).convert("RGB")
        mask_pil = Image.fromarray(mask_uint8).convert("L")

        img_pil_resized = img_pil.resize((self.image_size, self.image_size), Image.BICUBIC)
        mask_pil_resized = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        out_pil = self._pipe(
            img_pil_resized,
            mask_pil_resized,
            image_size=self.image_size,
            num_steps=self.num_inference_steps,
            strength=0.999,
            guidance_scale=self.guidance_scale,
            noise_offset=0.0357,
            paste=False,
        )[0]

        # Resize the network output back to the original resolution.
        if out_pil.size != (orig_w, orig_h):
            out_pil = out_pil.resize((orig_w, orig_h), Image.BICUBIC)
        out_np = np.asarray(out_pil.convert("RGB"), dtype=np.uint8)

        # Paste only the masked region over the original to avoid touching
        # pixels we didn't intend to edit.
        m = mask_uint8 > 127
        if m.any():
            result = image_rgb_uint8.copy()
            result[m] = out_np[m]
            return result
        return image_rgb_uint8.copy()
