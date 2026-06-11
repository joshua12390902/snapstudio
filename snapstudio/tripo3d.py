"""TripoSR 任意角度主角：單張去背產品照 → 重建 → render 任何仰角/方位。

解決「6 離散角度配不上場景」：可 render 場景/握持需要的**精確連續角度**，
該角度去背圖再進既有 inpaint 管線當鎖定錨點。純 torch NeRF render（不需
torchmcubes/nvcc/EGL；mesh 匯出已 stub，僅 render）。
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

from . import config

import numpy as np
import torch
from PIL import Image

_TRIPO_REPO = config.ROOT / "TripoSR"
if str(_TRIPO_REPO) not in sys.path:
    sys.path.insert(0, str(_TRIPO_REPO))

TRIPO_DIR = config.WEIGHTS / "triposr"


class TripoProduct:
    """TripoSR 包裝：reconstruct 一次，之後 render 任意角度。"""

    def __init__(self, device: str = "cuda", chunk: int = 8192):
        self.device = device
        self.model = None
        self.chunk = chunk
        self._matting = None

    def _load(self):
        from tsr.system import TSR
        m = TSR.from_pretrained(str(TRIPO_DIR), config_name="config.yaml",
                                weight_name="model.ckpt")
        m.renderer.set_chunk_size(self.chunk)
        self.model = m.to(self.device)

    @staticmethod
    def _to_gray_bg(rgba: Image.Image, margin: float = 1.3) -> Image.Image:
        """去背 RGBA → 置中 0.5 灰底 512 方形（TripoSR 慣例輸入）。"""
        bbox = rgba.convert("RGBA").split()[-1].getbbox()
        fg = rgba.crop(bbox) if bbox else rgba
        side = int(max(fg.size) * margin)
        cv = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        cv.paste(fg, ((side - fg.width) // 2, (side - fg.height) // 2), fg)
        a = np.array(cv.resize((512, 512), Image.LANCZOS)).astype(np.float32) / 255
        img = a[:, :, :3] * a[:, :, 3:4] + 0.5 * (1 - a[:, :, 3:4])
        return Image.fromarray((img * 255).astype(np.uint8))

    def reconstruct(self, rgba_cutout: Image.Image):
        if self.model is None:
            self._load()
        return self.model([self._to_gray_bg(rgba_cutout)], device=self.device)

    @torch.no_grad()
    def render_angle(self, scene_codes, elevation: float = 10.0,
                     azimuth: float = 0.0, size: int = 768,
                     n_sweep: int = 12) -> Image.Image:
        """render 指定 (仰角, 方位) 的產品圖，回傳 SAM2 重去背的 RGBA。

        以 n_sweep 個方位環繞 render，挑最接近目標方位者（連續角度近似）。
        """
        rends = self.model.render(
            scene_codes, n_views=n_sweep, elevation_deg=float(elevation),
            return_type="pil", height=size, width=size,
        )[0]
        idx = int(round((azimuth % 360) / (360.0 / n_sweep))) % n_sweep
        view = rends[idx].convert("RGB")
        if self._matting is None:
            from .matting import Matting
            self._matting = Matting(refine_sam=True)
        return self._matting.cutout(view)

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()
