"""IC-Light 重打光模組（diffusers 0.39 改寫，本專題技術核心）。

機制（poc/poc_iclight.py 已驗證後模組化）：
- SD1.5 UNet conv_in 換成 8 通道（fc：雜訊+前景 latent）或 12 通道（fbc：再加背景 latent）
- 權重 = SD1.5 原版 + IC-Light offset 逐 key 相加
- 攔截 unet.forward，每步把條件 latent 串接進輸入
- fp16 陷阱：新建 conv_in 預設 fp32，必須 .to(unet.dtype)
"""
import gc
import time
import warnings

import numpy as np
import safetensors.torch as sf
import torch
from PIL import Image
from diffusers import (
    DPMSolverMultistepScheduler,
    LCMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

from . import config

DEFAULT_NEG = "lowres, bad quality, blurry, watermark, text, logo, cluttered"

# mode → (conv_in 輸入通道數, offset 權重檔)
_MODE_SPEC = {
    "fc": (8, config.ICLIGHT_FC),
    "fbc": (12, config.ICLIGHT_FBC),
}


def _compose_gray(rgba: Image.Image, size: tuple[int, int]) -> Image.Image:
    """RGBA → IC-Light 要求的灰底(127) RGB，保持長寬比貼齊 size（不足補灰）。

    優先用 matting.Matting.compose_gray（單一實作來源）；matting 不可匯入時
    退用 POC 同款混合 + 直接 resize（可能輕微變形，僅為退路）。
    """
    try:
        from .matting import Matting

        return Matting.compose_gray(rgba, size).convert("RGB")
    except ImportError:
        arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32)
        alpha = arr[..., 3:4] / 255.0
        comp = arr[..., :3] * alpha + 127.0 * (1.0 - alpha)
        return Image.fromarray(comp.astype(np.uint8)).resize(size, Image.LANCZOS)


def _round64(x: float) -> int:
    """取最接近的 64 倍數（UNet 三層下採樣的安全尺寸）。"""
    return max(64, int(round(x / 64.0)) * 64)


class Relighter:
    """IC-Light fc/fbc 重打光器。模型常駐（不反覆載卸），mode 固定於建構時。

    quality 決定 relight() 未明示 hires/lcm 時的預設檔位：
    "fine"＝25 步 + 兩段式高清（1.5x img2img）；"fast"＝LCM-LoRA 8 步預覽。
    """

    def __init__(self, mode: str = "fbc", quality: str = "fine", device: str = "cuda"):
        if mode not in _MODE_SPEC:
            raise ValueError(f"mode 必須是 fc 或 fbc，收到 {mode!r}")
        if quality not in ("fast", "fine"):
            raise ValueError(f"quality 必須是 'fast' 或 'fine'，收到 {quality!r}")
        self.mode = mode
        self.quality = quality
        self.device = device
        in_ch, offset_path = _MODE_SPEC[mode]

        pipe = StableDiffusionPipeline.from_pretrained(
            str(config.SD15_DIR), torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False,
        )
        unet = pipe.unet

        # conv_in 4→8/12：前 4 通道沿用原權重，新增通道補零
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_ch, unet.conv_in.out_channels,
                unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4].copy_(unet.conv_in.weight)
            new_conv_in.bias.copy_(unet.conv_in.bias)
            unet.conv_in = new_conv_in.to(unet.dtype)  # fp16 陷阱：新層預設 fp32

        offset = sf.load_file(str(offset_path))
        origin = unet.state_dict()
        unet.load_state_dict(
            {k: origin[k] + offset[k].to(origin[k]) for k in origin}, strict=True
        )
        del offset, origin

        # 攔截 forward：每步把條件 latent（經 cross_attention_kwargs 走私進來）串接到輸入
        original_forward = unet.forward

        def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
            cak = kwargs.get("cross_attention_kwargs") or {}
            c = cak["concat_conds"].to(sample)
            c = torch.cat([c] * (sample.shape[0] // c.shape[0]), dim=0)  # 配合 CFG 的 2x batch
            kwargs["cross_attention_kwargs"] = {}
            return original_forward(
                torch.cat([sample, c], dim=1), timestep, encoder_hidden_states, **kwargs
            )

        unet.forward = hooked_forward

        self._orig_sched_config = pipe.scheduler.config
        self._base_sched = DPMSolverMultistepScheduler.from_config(
            self._orig_sched_config, use_karras_sigmas=True
        )
        self._lcm_sched = None
        self._lcm_loaded = False
        pipe.scheduler = self._base_sched
        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        self.pipe = pipe

        # hires 第二段：img2img 共用同一套元件（不另佔 VRAM）
        self.pipe_i2i = StableDiffusionImg2ImgPipeline(
            **pipe.components, requires_safety_checker=False
        )
        self.pipe_i2i.set_progress_bar_config(disable=True)

    # ---------- 內部工具 ----------

    def _encode(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL → VAE latent（取分布眾數，乘 scaling factor）。"""
        t = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 127.5 - 1.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device, self.pipe.vae.dtype)
        return self.pipe.vae.encode(t).latent_dist.mode() * self.pipe.vae.config.scaling_factor

    def _concat_conds(self, fg: Image.Image, bg: Image.Image | None,
                      size: tuple[int, int]) -> torch.Tensor:
        """依 mode 組出條件 latent；fbc 為 cat([fg, bg], dim=1)。"""
        with torch.inference_mode():
            conds = self._encode(fg.resize(size, Image.LANCZOS))
            if self.mode == "fbc":
                conds = torch.cat(
                    [conds, self._encode(bg.resize(size, Image.LANCZOS))], dim=1
                )
        return conds

    def _ensure_lcm(self):
        if not self._lcm_loaded:
            # HF_HUB_OFFLINE=1 下從本地目錄載入必須指定 weight_name
            self.pipe.load_lora_weights(
                str(config.LCM_LORA_DIR),
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="lcm",
            )
            self._lcm_sched = LCMScheduler.from_config(self._orig_sched_config)
            self._lcm_loaded = True

    def _set_sched(self, sched):
        self.pipe.scheduler = sched
        self.pipe_i2i.scheduler = sched  # 兩條 pipeline 共用元件但 scheduler 屬性各自持有

    # ---------- 對外 API ----------

    def relight(self, foreground: Image.Image,
                background: Image.Image | None = None,
                prompt: str = "", *,
                negative_prompt: str = DEFAULT_NEG,
                width: int = 768, height: int = 768,
                steps: int = 25, cfg: float = 2.0,
                seed: int | None = None,
                hires: bool | None = None,
                lcm: bool | None = None) -> Image.Image:
        """重打光主流程（簽名依 pipeline.py 整合契約：fg, bg, prompt, …）。

        foreground：RGBA（自動轉灰底）或已是灰底 RGB。
        background：fbc 模式必填，會 resize 至同尺寸；fc 模式忽略。
        hires：兩段式（官方做法）：先 width×height 初稿 → 1.5x 放大 → img2img(strength 0.5)。
        lcm：掛 LCM-LoRA + LCMScheduler 快速檔；steps/cfg 若仍為預設值則自動改 8/1.5。
        hires/lcm 留 None 時依 self.quality 決定：fine → hires；fast → lcm。
        """
        if hires is None:
            hires = self.quality == "fine"
        if lcm is None:
            lcm = self.quality == "fast"
        width, height = _round64(width), _round64(height)
        negative = negative_prompt or DEFAULT_NEG
        if self.mode == "fbc":
            if background is None:
                raise ValueError("fbc 模式需要 background 圖")
            background = background.convert("RGB")
        elif background is not None:
            warnings.warn("fc 模式不使用 background，已忽略")
            background = None

        if foreground.mode != "RGB":
            gray = _compose_gray(foreground, (width, height))
        else:
            gray = foreground
        seed = int(seed) if seed is not None else int(torch.seed() % (2 ** 31))

        if lcm:
            self._ensure_lcm()
            self.pipe.enable_lora()
            self._set_sched(self._lcm_sched)
            if steps == 25:
                steps = 8
            if cfg == 2.0:
                cfg = 1.5
        else:
            if self._lcm_loaded:
                self.pipe.disable_lora()
            self._set_sched(self._base_sched)

        conds = self._concat_conds(gray, background, (width, height))
        img = self.pipe(
            prompt=prompt, negative_prompt=negative,
            width=width, height=height,
            num_inference_steps=steps, guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": conds},
            generator=torch.Generator(self.device).manual_seed(seed),
        ).images[0]

        if hires:
            hw, hh = _round64(width * 1.5), _round64(height * 1.5)
            conds_hi = self._concat_conds(gray, background, (hw, hh))
            img = self.pipe_i2i(
                prompt=prompt, negative_prompt=negative,
                image=img.resize((hw, hh), Image.LANCZOS), strength=0.5,
                num_inference_steps=steps, guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": conds_hi},
                generator=torch.Generator(self.device).manual_seed(seed),
            ).images[0]
        return img

    def unload(self):
        """釋放 GPU（共用卡禮儀）。"""
        self.pipe_i2i = None
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()


# ---------- 自測：python -m snapstudio.relight ----------
if __name__ == "__main__":
    out_dir = config.EXAMPLES / "dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    fg_rgba = Image.open(config.EXAMPLES / "test_cutout.png")  # POC 產出的 RGBA 去背圖

    def bench(name, fn):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        img = fn()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        img.save(out_dir / name)
        print(f"[{name}] {dt:.1f}s | peak VRAM {peak:.2f}GiB", flush=True)

    prompt_fc = ("brown leather wallet on dark walnut table, warm morning window light "
                 "from left side, cozy cafe atmosphere, soft shadows, "
                 "professional product photography, best quality")
    r = Relighter("fc")
    print("[fc loaded]", flush=True)
    bench("relight_fc.png",
          lambda: r.relight(fg_rgba, prompt=prompt_fc, seed=42, hires=False, lcm=False))
    r.unload()
    del r
    gc.collect()
    torch.cuda.empty_cache()

    # 純色漸層背景：驗證 fbc 串接機制（正式背景由 scene.py 供應）
    ramp = np.linspace(0.0, 1.0, 768, dtype=np.float32)[:, None, None]
    top, bottom = np.array([255, 205, 150], np.float32), np.array([45, 28, 18], np.float32)
    bg = Image.fromarray((top * (1 - ramp) + bottom * ramp).astype(np.uint8)
                         .repeat(768, axis=1))
    bg.save(out_dir / "relight_test_bg_gradient.png")

    prompt_fbc = ("brown leather wallet, warm ambient glow from above, soft shadows, "
                  "professional product photography, best quality")
    r = Relighter("fbc")
    print("[fbc loaded]", flush=True)
    bench("relight_fbc.png",
          lambda: r.relight(fg_rgba, bg, prompt_fbc, seed=42, hires=False, lcm=False))
    bench("relight_fbc_hires.png",
          lambda: r.relight(fg_rgba, bg, prompt_fbc, seed=42, hires=True, lcm=False))
    bench("relight_fbc_lcm.png",
          lambda: r.relight(fg_rgba, bg, prompt_fbc, seed=42, hires=False, lcm=True))
    r.unload()
    print("DONE", flush=True)
