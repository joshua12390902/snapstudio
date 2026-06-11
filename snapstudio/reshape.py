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
    return fill, c  # 回傳輪廓供 fitEllipse 求幾何中心（對日期窗/反光等不對稱穩健）


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
        real_fill, real_cnt = r
        gen_fill, gen_cnt = g
        Hh, Ww = gen_arr.shape[:2]

        def _center_radius(cnt, fill):
            # 幾何橢圓中心（對日期窗缺口/反光不對稱穩健），點太少退回質心
            if len(cnt) >= 5:
                (cx, cy), (MA, ma), _ = cv2.fitEllipse(cnt)
                return cx, cy, (MA + ma) / 4.0
            m = cv2.moments(fill, binaryImage=True)
            if m["m00"] < 1:
                return None
            return (m["m10"] / m["m00"], m["m01"] / m["m00"], (m["m00"] / np.pi) ** 0.5)
        rc, gc = _center_radius(real_cnt, real_fill), _center_radius(gen_cnt, gen_fill)
        if rc is None or gc is None:
            return gen_img
        rcx, rcy, rrad = rc
        gcx, gcy, grad = gc
        # 真實錶盤略放大過蓋（1.08），完全蓋住生成錶盤、消除邊緣殘色；由凸包遮罩裁邊不溢出
        scale = grad / max(rrad, 1.0) * 1.08

        def _affine(angle_deg):
            a = np.deg2rad(angle_deg)
            ca, sa = np.cos(a) * scale, np.sin(a) * scale
            # 真實點 p → 繞真實質心旋轉縮放 → 平移到生成質心（剛性，圓盤精準對位）
            return np.array([[ca, -sa, gcx - (ca * rcx - sa * rcy)],
                             [sa, ca, gcy - (sa * rcx + ca * rcy)]], dtype=np.float32)

        # 邊緣相關旋轉搜尋：對「刻度/指針/錶框」邊緣找最佳角度（不靠亂碼文字）
        gen_edge = cv2.Canny(cv2.cvtColor(gen_arr, cv2.COLOR_RGB2GRAY), 60, 160).astype(np.float32)
        gen_edge[gen_fill == 0] = 0
        real_edge = cv2.Canny(cv2.cvtColor(real_arr, cv2.COLOR_RGB2GRAY), 60, 160).astype(np.float32)
        best = (-2.0, 0.0, None)
        for ang in range(-24, 25, 3):
            M = _affine(ang)
            we = cv2.warpAffine(real_edge, M, (Ww, Hh))
            wf = cv2.warpAffine(real_fill, M, (Ww, Hh))
            sel = (gen_fill > 0) & (wf > 0)
            if sel.sum() < 200:
                continue
            a_, b_ = gen_edge[sel], we[sel]
            if a_.std() < 1e-3 or b_.std() < 1e-3:
                continue
            ncc = float(np.corrcoef(a_, b_)[0, 1])
            if ncc > best[0]:
                best = (ncc, float(ang), M)
        ncc, ang, M = best
        if M is None or ncc < 0.05:   # 對不上→正立置中（不硬轉）
            ang, M = 0.0, _affine(0.0)
        logger.info("composite_real_face angle=%.0f ncc=%.3f", ang, ncc)
        warped = cv2.warpAffine(real_arr, M, (Ww, Hh), flags=cv2.INTER_LINEAR)
        wfill = cv2.warpAffine(real_fill, M, (Ww, Hh), flags=cv2.INTER_NEAREST)
        region = cv2.erode(cv2.bitwise_and(gen_fill, wfill), np.ones((3, 3), np.uint8))
        if region.sum() < 500:
            return gen_img
        alpha = Image.fromarray(region).filter(ImageFilter.GaussianBlur(2.5))
        return Image.composite(Image.fromarray(warped), gen_img.convert("RGB"), alpha)
    except Exception as exc:  # noqa: BLE001  # 合成失敗絕不可拖垮主流程
        logger.warning("composite_real_face 失敗，用原圖：%s", exc)
        return gen_img


def compact_reference(real_rgba: Image.Image, margin: float = 1.9) -> Image.Image:
    """為 AnyDoor 準備「精簡參考圖」：裁到產品主要平面面（錶盤/標籤）+ 邊距。

    為何：AnyDoor 對「主體＋拖尾」形狀（如折疊收納的手錶＝錶頭＋折疊錶帶）會把拖尾
    的扣環誤生成第二個圓盤＝雙錶（實測過）。裁到緊湊的主面，AnyDoor 才會擺出單一乾淨
    產品、錶帶由它自然環繞生成；真實細節仍由 composite_real_face 用完整原圖補回。
    偵測不到主面就退回原圖（不破壞）。沿用 composite 的同一套色彩自適應面偵測，可泛化。"""
    try:
        import cv2  # noqa: F401
        import numpy as np
        rgba = real_rgba.convert("RGBA")
        ab = rgba.split()[-1].getbbox()
        if ab:
            rgba = rgba.crop(ab)          # 先裁到產品本體，後續偵測/裁切同一座標系
        arr = np.array(_on_white(rgba))   # _on_white 內部 bbox 已滿版，尺寸與 rgba 一致
        hue = _dominant_hue(arr)
        if hue is None:
            return rgba
        fr = _face_region(arr, hue)
        if fr is None:
            return rgba
        fill, _cnt = fr
        ys, xs = np.where(fill > 0)
        if len(xs) < 50:
            return rgba
        cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 * margin
        W, H = rgba.size
        box = (max(0, int(cx - half)), max(0, int(cy - half)),
               min(W, int(cx + half)), min(H, int(cy + half)))
        crop = rgba.crop(box)
        ab2 = crop.split()[-1].getbbox()  # 收緊到 alpha，去掉裁框內殘留透明邊
        return crop.crop(ab2) if ab2 else crop
    except Exception:  # noqa: BLE001
        return real_rgba.convert("RGBA")


def framing_for(card, product_class: str) -> str:
    """取景片語「戴/握在哪」：優先用 VLM 決定的 card.worn_framing（交給 LLM 判斷），
    LLM 沒給時才用一行通用備援。不再硬寫品類對照表。"""
    wf = (getattr(card, "worn_framing", "") or "").strip()
    if wf:
        return wf
    if product_class == "handheld":
        return "a hand holding the product naturally"
    # wearable 但 VLM 沒給部位：退回「手腕特寫」（最常見穿戴情境），
    # 絕不可退回「a person」——那會讓 bare_scene_prompt 生出人臉肖像、產品被貼到臉上。
    return "a person's bare forearm and wrist, the wrist clearly visible in close-up"


# 裸場景負面詞：身體部位要「乾淨裸露」給 AnyDoor 擺產品，所以排除任何既有的
# 穿戴品（手錶/首飾…）與人臉肖像。放 negative_prompt（不會被 CLIP 77-token 截掉），
# 不塞正面 prompt（實測塞正面會被截斷而失效，場景反而自己長出一支錶）。
BARE_SCENE_NEGATIVE = ("watch, wristwatch, smartwatch, jewelry, bracelet, bangle, ring, "
                       "glasses, necklace, face, portrait, headshot, full body, "
                       "multiple people, crowd, text, logo")


def bare_scene_prompt(framing: str, scene_context: str = "", category: str = "") -> str:
    """把 framing（戴/握的取景）轉成「裸身體部位」場景 prompt，供 AnyDoor 當擺放目標。

    鐵則：身體部位必須是**畫面主體**；行銷情境（scene_context）只當背景/氛圍，否則
    text2img 會生出人臉肖像（實測過），AnyDoor 便把產品貼到臉上。身體部位關鍵字擺
    **最前面**，確保即使 CLIP 77-token 截斷也保得住主體；「不要既有手錶/人臉」等負面
    交給 BARE_SCENE_NEGATIVE。例：'a person's wrist...' → 乾淨手腕特寫、留白給 AnyDoor。"""
    import re
    bare = re.sub(r"\b(wearing|holding)\b.*", "", framing, flags=re.I).strip().rstrip(", ")
    if not bare:
        bare = "a person's bare forearm and wrist"
    # 手持類：裸場景若主體是「手」，改成「攤開放鬆、掌心朝上呈現」——避免生出
    # 「握著空氣」的詭異手勢（實測 controller 兩手空抓很恐怖），讓 AnyDoor 把產品擺在
    # 攤開的掌上才自然。但穿戴部位（耳/腕/指/頸）優先保留，不可被 hand 規則蓋掉。
    low = bare.lower()
    if not any(k in low for k in ("ear", "wrist", "forearm", "finger", "neck", "face", "head")):
        if "hand" in low or "palm" in low:
            both = ("two" in low) or ("both" in low) or ("hands" in low)
            hands = "two open relaxed hands side by side" if both else "an open relaxed hand"
            bare = f"{hands}, palm facing up, fingers relaxed and open, gently presenting"
    ctx = f", {scene_context} blurred in the background" if scene_context else ""
    return (f"close-up photo of {bare}, the body part fills the frame, bare skin, "
            f"angled toward the camera{ctx}, soft even light, photorealistic, high detail")


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

    @torch.no_grad()
    def generate_scene(self, prompt: str, *, negative_prompt: str = "",
                       seed: int | None = None, steps: int = 30,
                       guidance_scale: float = 6.0, width: int = 1024,
                       height: int = 768) -> Image.Image:
        """純 text2img 生成「裸身體部位」場景（供 AnyDoor 當擺放目標）。
        IP-Adapter 暫設 scale=0 不作用、餵白圖佔位，等同純 text2img。"""
        if self.pipe is None:
            self._load()
        self.pipe.set_ip_adapter_scale(0.0)
        blank = Image.new("RGB", (224, 224), (255, 255, 255))
        neg = f"{negative_prompt}, {RESHAPE_NEGATIVE}" if negative_prompt else RESHAPE_NEGATIVE
        gen = (torch.Generator(self.device).manual_seed(int(seed))
               if seed is not None else None)
        img = self.pipe(
            prompt=prompt, negative_prompt=neg, ip_adapter_image=blank,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            width=width, height=height, generator=gen,
        ).images[0]
        self.pipe.set_ip_adapter_scale(self.ip_scale)  # 還原
        return img

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
