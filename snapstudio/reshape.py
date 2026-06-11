"""重塑模式（reshape / worn-held）：用 IP-Adapter 保留產品「身份特徵」，
讓 SDXL 重畫產品的姿態（戴在手腕/身上、握在手中），而非鎖死像素。

與鎖定模式（groundgen.SceneInpainter）的分工：
- 剛性產品（香水/罐/皮夾）→ 鎖定模式：像素級精準、站桌上。
- 穿戴/手持產品（手錶/戒指/手把）→ 本模組：可重塑姿態、有故事，代價是
  產品表面文字/logo 不保證精準（生成必然的取捨）。

技術（POC 驗證）：RealVisXL V5 text2img + ip-adapter-plus_sdxl_vit-h；
ip_adapter_scale≈0.5 是甜蜜點（太高 image prompt 壓過文字＝沒戴上身）。
參考圖用去背後貼白底的產品（避免凌亂背景汙染身份特徵）。
"""
from __future__ import annotations

import gc
import logging

from . import config  # 先匯入，套用 HF_HUB_OFFLINE

import torch
from PIL import Image, ImageFilter
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logger = logging.getLogger(__name__)

IP_DIR = config.WEIGHTS / "ip-adapter"

# 重塑模式共用負面詞：擋掉「沒戴上身/平放桌面」與手部崩壞
RESHAPE_NEGATIVE = (
    "product alone on table, floating product, no person, no hand, no wrist, "
    "deformed hands, extra fingers, fused fingers, missing fingers, mutated, "
    "blurry, lowres, plastic toy, distorted, melted, watermark, duplicate, "
    "second product, oversaturated"
)

def _face_region(rgb, hue: int, tol: int = 20):
    """依主色 hue 抓「平面細節面」（錶盤/標籤）→ 回 (填滿遮罩, 軸對齊bbox x,y,w,h)。
    用軸對齊 bbox（非旋轉矩形）→ 真實面縮放貼回時保持正立，不會被轉歪。"""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lo = (max(0, hue - tol), 55, 25)
    hi = (min(179, hue + tol), 255, 240)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    h, w = rgb.shape[:2]
    if cv2.contourArea(c) < 0.008 * h * w:  # 太小，視為沒找到
        return None
    fill = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(fill, cv2.convexHull(c), 255)
    return fill, cv2.boundingRect(c)


def _dominant_hue(rgb):
    """真實產品最強飽和色的 hue（錶盤/標籤主色）；無明顯彩色回 None。"""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sel = (s > 70) & (v > 35) & (v < 240)
    if sel.sum() < 200:
        return None
    return int(np.argmax(np.bincount(h[sel].ravel(), minlength=180)))


def composite_real_face(gen_img: Image.Image, real_rgba: Image.Image) -> Image.Image:
    """混合法：把真實產品的「平面細節面」（如錶盤）homography 變形貼回重塑成品，
    補回 IP-Adapter 重塑丟失的小細節（logo/文字/刻度）。色彩自適應、最佳努力，
    偵測不到就原樣回傳（不破壞）。"""
    try:
        import cv2
        import numpy as np
        real_rgb = _on_white(real_rgba)
        gen_arr = np.array(gen_img.convert("RGB"))
        real_arr = np.array(real_rgb)
        hue = _dominant_hue(real_arr)
        if hue is None:
            return gen_img
        r = _face_region(real_arr, hue)
        g = _face_region(gen_arr, hue)
        if r is None or g is None:
            return gen_img
        real_fill, (rx, ry, rw, rh) = r
        gen_fill, (gx, gy, gw, gh) = g
        Hh, Ww = gen_arr.shape[:2]
        # 1) 軸對齊初始：真實錶盤裁切→縮到生成錶盤 bbox→放到全圖對應位置
        crop = real_arr[ry:ry + rh, rx:rx + rw]
        cropf = real_fill[ry:ry + rh, rx:rx + rw]
        init = np.zeros_like(gen_arr)
        init[gy:gy + gh, gx:gx + gw] = cv2.resize(crop, (gw, gh), interpolation=cv2.INTER_AREA)
        initf = np.zeros((Hh, Ww), np.uint8)
        initf[gy:gy + gh, gx:gx + gw] = cv2.resize(cropf, (gw, gh), interpolation=cv2.INTER_NEAREST)
        # 2) ECC 剛性旋轉精修：把真實錶盤對齊「生成錶盤實際朝向」（指針/刻度/日期窗結構），
        #    只旋轉+平移不變形。cc=相關係數=檢查機制，太低代表對不上→退回軸對齊不硬轉。
        warp = np.eye(2, 3, dtype=np.float32)
        cc = -1.0
        try:
            tmpl = cv2.GaussianBlur(cv2.cvtColor(gen_arr, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
            inp = cv2.GaussianBlur(cv2.cvtColor(init, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-5)
            cc, warp = cv2.findTransformECC(tmpl, inp, warp, cv2.MOTION_EUCLIDEAN, crit, initf, 5)
        except cv2.error:
            cc = -1.0
        if cc >= 0.45:
            aligned = cv2.warpAffine(init, warp, (Ww, Hh), flags=cv2.INTER_LINEAR)
            af = cv2.warpAffine(initf, warp, (Ww, Hh), flags=cv2.INTER_NEAREST)
        else:
            aligned, af = init, initf  # 對不上→不硬轉，維持軸對齊
        logger.info("composite_real_face ECC cc=%.3f", cc)
        region = cv2.erode(cv2.bitwise_and(gen_fill, af), np.ones((3, 3), np.uint8))
        if region.sum() < 500:
            return gen_img
        alpha = Image.fromarray(region).filter(ImageFilter.GaussianBlur(2.5))
        return Image.composite(Image.fromarray(aligned), gen_img.convert("RGB"), alpha)
    except Exception as exc:  # noqa: BLE001  # 合成失敗絕不可拖垮主流程
        logger.warning("composite_real_face 失敗，用原圖：%s", exc)
        return gen_img


def framing_for(card, product_class: str) -> str:
    """取景片語「戴/握在哪」：優先用 VLM 決定的 card.worn_framing（交給 LLM 判斷），
    LLM 沒給時才用一行通用備援。不再硬寫品類對照表。"""
    wf = (getattr(card, "worn_framing", "") or "").strip()
    if wf:
        return wf
    if product_class == "handheld":
        return "a hand holding the product naturally"
    return "a person wearing the product, the body part clearly visible"


def _on_white(rgba: Image.Image) -> Image.Image:
    """去背 RGBA → 貼白底 RGB（IP-Adapter 參考圖；白底避免雜訊汙染身份）。"""
    rgba = rgba.convert("RGBA")
    bbox = rgba.split()[-1].getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, (0, 0), rgba)
    return bg


class ReshapeStudio:
    """IP-Adapter 重塑生成器。模型常駐；切到鎖定模式前可 unload 讓 VRAM。"""

    def __init__(self, device: str = "cuda", ip_scale: float = 0.5):
        self.device = device
        self.ip_scale = ip_scale
        self.pipe: StableDiffusionXLPipeline | None = None

    def _load(self) -> None:
        vae = AutoencoderKL.from_pretrained(config.SDXL_VAE_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(config.REALVISXL_PATH), torch_dtype=torch.float16, vae=vae)
        # 預載 image_encoder（離線），再掛 IP-Adapter 權重
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(IP_DIR / "models" / "image_encoder"), torch_dtype=torch.float16)
        pipe.image_encoder = image_encoder
        pipe.feature_extractor = CLIPImageProcessor()
        pipe.load_ip_adapter(
            str(IP_DIR), subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
            image_encoder_folder=None)
        pipe.set_ip_adapter_scale(self.ip_scale)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)
        logger.info("ReshapeStudio 載入完成（ip_scale=%s）", self.ip_scale)

    @torch.no_grad()
    def generate(
        self,
        product_rgba: Image.Image,
        framing: str,
        scene_context: str,
        *,
        negative_prompt: str = "",
        steps: int = 32,
        guidance_scale: float = 6.5,
        seed: int | None = None,
        ip_scale: float | None = None,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """重塑生成：把產品（身份）重畫成 framing 描述的穿戴/手持姿態 + 場景。

        framing：取景片語（戴在手腕/握在手中…），寫在 prompt 最前（CLIP 重前段）。
        scene_context：背景＋光線氛圍（來自 plan_scenes）。
        """
        if self.pipe is None:
            self._load()
        if ip_scale is not None:
            self.pipe.set_ip_adapter_scale(ip_scale)
        ref = _on_white(product_rgba)
        prompt = (
            f"professional advertising photo of {framing}, {scene_context}, "
            "cinematic lighting, shallow depth of field, high detail, commercial photography"
        )
        neg = f"{negative_prompt}, {RESHAPE_NEGATIVE}" if negative_prompt else RESHAPE_NEGATIVE
        gen = (torch.Generator(self.device).manual_seed(int(seed))
               if seed is not None else None)
        img = self.pipe(
            prompt=prompt, negative_prompt=neg, ip_adapter_image=ref,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            width=width, height=height, generator=gen,
        ).images[0]
        return img

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
