"""場景背景生成：RealVisXL V5.0（SDXL 寫實微調）依場景企劃 prompt 產生商品攝影背景。

設計重點：
- 模型常駐 GPU（以 VRAM 換品質），由 pipeline.py 統一呼叫 unload() 釋放
- RealVisXL 單檔權重離線載入失敗時自動退回 SDXL base（HF 快取已有）
- VAE 一律換 madebyollin/sdxl-vae-fp16-fix，避免 fp16 下溢位出黑圖
"""
from __future__ import annotations

import gc
import logging

from . import config  # 須先於 diffusers 載入，套用 HF_HUB_OFFLINE 等環境變數

import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)

logger = logging.getLogger(__name__)

REALVISXL_PATH = config.REALVISXL_PATH

# 商品攝影背景的基線負面詞；LLM 場景企劃給的 negative 會與此合併
BASE_NEGATIVE = (
    "lowres, bad quality, blurry, jpeg artifacts, text, watermark, logo, "
    "people, hands, deformed, cluttered"
)
# RealVisXL 官方建議 CFG 5-7；過高易出油畫感
GUIDANCE_SCALE = 6.0


class SceneGenerator:
    """SDXL 場景背景生成器（預設 RealVisXL V5.0，失敗退 SDXL base）。"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe: StableDiffusionXLPipeline | None = None
        self.model_name: str | None = None
        self._load()

    def _load(self) -> None:
        vae = AutoencoderKL.from_pretrained(
            config.SDXL_VAE_ID, torch_dtype=torch.float16
        )
        try:
            if not REALVISXL_PATH.exists():
                raise FileNotFoundError(REALVISXL_PATH)
            # 單檔權重借 SDXL base 快取的 config/tokenizer 離線組裝
            pipe = StableDiffusionXLPipeline.from_single_file(
                str(REALVISXL_PATH),
                torch_dtype=torch.float16,
                config=config.SDXL_ID,
                local_files_only=True,
            )
            self.model_name = "RealVisXL_V5.0"
        except Exception as exc:
            logger.warning("RealVisXL 載入失敗（%s），退回 SDXL base", exc)
            pipe = StableDiffusionXLPipeline.from_pretrained(
                config.SDXL_ID, torch_dtype=torch.float16, variant="fp16"
            )
            self.model_name = "sdxl-base-1.0"
        pipe.vae = vae
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)
        logger.info("SceneGenerator 載入完成：%s", self.model_name)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        seed: int | None = None,
    ) -> Image.Image:
        """依場景 prompt 生成一張背景圖；negative_prompt 與基線負面詞合併後送入管線。"""
        if self.pipe is None:
            self._load()  # unload() 後再次使用時自動重載
        neg = f"{negative_prompt}, {BASE_NEGATIVE}" if negative_prompt else BASE_NEGATIVE
        gen = (
            torch.Generator(self.device).manual_seed(int(seed))
            if seed is not None
            else None
        )
        return self.pipe(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=gen,
        ).images[0]

    def generate_batch(self, prompts: list[str], **kw) -> list[Image.Image]:
        """多組場景逐張生成（介面預留，之後可改真批次）。"""
        return [self.generate(p, **kw) for p in prompts]

    def unload(self) -> None:
        """釋放 GPU 記憶體；之後再呼叫 generate() 會自動重載。"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
