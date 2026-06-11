"""多視角主角：單張去背產品照 → Zero123++ 6 視角 → SAM2 重去背 → 可旋轉的主角資產。

讓產品從「被鎖死的 2D 貼紙」變成「可選角度的主角」：使用者/情境可挑一個視角，
該視角去背圖再進既有 inpaint 管線當鎖定錨點生成場景（含握持情境的角度來源）。

研究/實測定位：Zero123++ 適合「多角度選圖 + 握持合成」MVP（view 正面/傾斜可用，
背面模型腦補較糊）；要平滑 360°/任意俯仰再上 SF3D 真 mesh。
"""
from __future__ import annotations

import gc
import logging

from . import config  # 先於 diffusers 匯入

import torch
from PIL import Image

logger = logging.getLogger(__name__)

Z123_DIR = config.WEIGHTS / "zero123plus"
Z123_PIPE = config.WEIGHTS / "zero123plus_pipe"
# Zero123++ v1.2 的 6 個固定方位（檢查員實測：4=正面最佳、0=傾斜次之可當主角）
USABLE_VIEWS = [4, 0, 5]  # 對外預設挑這幾個（背面 1/2/3 糊，隱藏）


class MultiViewGenerator:
    """Zero123++ 包裝：cutout → 6 視角 RGBA（已 SAM2 重去背）。"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        self._matting = None

    def _load(self):
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        pipe = DiffusionPipeline.from_pretrained(
            str(Z123_DIR), custom_pipeline=str(Z123_PIPE),
            torch_dtype=torch.float16, local_files_only=True, trust_remote_code=True,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)

    @staticmethod
    def _prep(rgba: Image.Image) -> Image.Image:
        """去背 RGBA → 置中白底 512 方形（Zero123++ 要求單物件白底方形）。"""
        bbox = rgba.convert("RGBA").split()[-1].getbbox()
        fg = rgba.crop(bbox) if bbox else rgba
        side = int(max(fg.size) * 1.4)
        canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
        canvas.paste(fg, ((side - fg.width) // 2, (side - fg.height) // 2), fg)
        return canvas.convert("RGB").resize((512, 512), Image.LANCZOS)

    @torch.no_grad()
    def six_views(self, rgba_cutout: Image.Image, steps: int = 36,
                  upscale: int = 768) -> list[Image.Image]:
        """回傳 6 個視角的 RGBA 去背圖（已 SAM2 重去背、放大到 upscale 方形）。"""
        if self.pipe is None:
            self._load()
        grid = self.pipe(self._prep(rgba_cutout), num_inference_steps=steps).images[0]
        gw, gh = grid.size
        cw, ch = gw // 2, gh // 3
        if self._matting is None:
            from .matting import Matting
            self._matting = Matting(refine_sam=True)
        views = []
        for i in range(6):
            r, c = i // 2, i % 2
            tile = grid.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
            tile = tile.resize((upscale, upscale), Image.LANCZOS)
            views.append(self._matting.cutout(tile))  # 重去背成透明
        return views

    def best_views(self, rgba_cutout: Image.Image, **kw) -> list[Image.Image]:
        """只回傳實測可用的視角（正面/傾斜/乾淨側面），供 UI 採視角列。"""
        allv = self.six_views(rgba_cutout, **kw)
        return [allv[i] for i in USABLE_VIEWS]

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
