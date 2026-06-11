"""Inpaint-grounded 場景生成：鎖住產品像素，讓 SDXL inpaint 生成產品「周圍」的
場景與檯面，接地陰影、反光在同一次去噪中自然長出（取代舊的「先造背景再貼產品」）。

關鍵（研究查證）：
- 必須用專用 9 通道 inpaint 權重（diffusers/stable-diffusion-xl-1.0-inpainting-0.1），
  RealVisXL 是 4 通道 base，塞不進 inpaint conv_in，硬塞會看到 mask 輪廓＝貼紙感
- LCM-LoRA-SDXL 可配 inpaint pipeline（HF 官方 demo 背書）：4 步、GS=1.0、LCMScheduler
- strength=0.99（非 1.0，1.0 會引雜訊掉畫質）、padding_mask_crop 提升邊界細節
- mask blur 羽化過渡；diffusers #10690 有二值化 bug，載入後驗證 mask 真為灰階漸層
"""
from __future__ import annotations

import gc
import logging

from . import config  # 先於 diffusers 匯入，套用 HF_HUB_OFFLINE

import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    StableDiffusionXLInpaintPipeline,
)

logger = logging.getLogger(__name__)

# 精簡高衝擊版（CLIP 只讀前 77 token，過長後段失效）：只放「每張都該擋」的
# 通用致命項；產品特定負面詞(barrel can/money clip/phantom subdial…)由 LLM 依
# 品類在 per-job negative 補。整夜 1-3 批驗證最高頻硬傷 + NSFW 合規。
BASE_NEGATIVE = (
    "lowres, blurry, low quality, deformed, distorted, floating, no contact shadow, "
    "cutout halo, sticker edge, jagged edge, duplicate product, second product, "
    "extra object, clutter, gibberish text, watermark, oversaturated, oil painting, "
    "plastic look, melted, mushy, conflicting shadows, visible seam, nsfw, nudity, "
    "exposed skin, deformed hands"
)
# 純背景（電商）模式才加的負面詞：擋掉人物與雜物，逼出乾淨商品攝影
NO_PEOPLE_NEGATIVE = "people, person, man, woman, hands, crowd, extra objects, clutter"


class SceneInpainter:
    """SDXL inpaint-grounded 場景生成器。模型常駐；accelerate=True 掛 LCM-LoRA 走 4 步。"""

    def __init__(self, device: str = "cuda", accelerate: bool = True,
                 model_dir=None):
        self.device = device
        self.accelerate = accelerate
        # 預設用 config.INPAINT_DIR（RealVisXL V4 inpaint 存在則優先，美感較佳）
        self.model_dir = str(model_dir) if model_dir else str(config.INPAINT_DIR)
        self.pipe: StableDiffusionXLInpaintPipeline | None = None
        self._load()

    def _load(self) -> None:
        vae = AutoencoderKL.from_pretrained(config.SDXL_VAE_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16, variant="fp16",
            local_files_only=True,
        )
        pipe.vae = vae
        if self.accelerate:
            # LCM：少步加速；fuse 後權重併入 UNet，之後不再切 adapter
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(
                str(config.LCM_LORA_SDXL_DIR),
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="lcm",
            )
            pipe.fuse_lora()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)
        logger.info("SceneInpainter 載入完成（accelerate=%s）", self.accelerate)

    @torch.no_grad()
    def generate(
        self,
        init: Image.Image,
        mask: Image.Image,
        prompt: str,
        *,
        negative_prompt: str = "",
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        blur_factor: int = 8,
        padding_mask_crop: int | None = 32,
        allow_people: bool = False,
    ) -> Image.Image:
        """在 mask 白區生成場景；init 含已定位的產品與接地陰影，產品（mask 黑區）保留。

        steps/guidance 留 None 時依 accelerate 自動：LCM 4 步 GS=1.0 / 標準 28 步 GS=6.0。
        allow_people=False（純背景）時加擋人物的負面詞；True（情境照）時允許人物/模特。
        """
        if self.pipe is None:
            self._load()
        if steps is None:
            steps = 4 if self.accelerate else 28
        if guidance_scale is None:
            guidance_scale = 1.0 if self.accelerate else 6.0
        neg = f"{negative_prompt}, {BASE_NEGATIVE}" if negative_prompt else BASE_NEGATIVE
        if not allow_people:
            neg = f"{NO_PEOPLE_NEGATIVE}, {neg}"
        w, h = init.size

        # 羽化 mask 邊界；驗證沒中 #10690 二值化 bug（極差→警告，但仍續跑）
        blurred = self.pipe.mask_processor.blur(mask, blur_factor=blur_factor)
        extrema = blurred.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            logger.warning("mask blur 後為純色（可能中 #10690），改用原始 mask")
            blurred = mask

        gen = (torch.Generator(self.device).manual_seed(int(seed))
               if seed is not None else None)
        kw = dict(
            prompt=prompt, negative_prompt=neg,
            image=init, mask_image=blurred,
            width=w, height=h,
            strength=0.99, num_inference_steps=steps,
            guidance_scale=guidance_scale, generator=gen,
        )
        if padding_mask_crop is not None:
            kw["padding_mask_crop"] = padding_mask_crop
        return self.pipe(**kw).images[0]

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
