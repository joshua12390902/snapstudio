"""Differential Diffusion 混合法 POC：把真實錶頭(殼+盤)貼到生成手腕的錶位置，
用 per-pixel 變更量 map 軟凍結錶頭(真實像素不動)、只和諧化接縫/手腕/光照。

驗證：錶盤=真實像素(TISSOT 不亂碼)、朝向/置中由我控制、接縫自然融合。
用法：PYTHONPATH=. python poc/worn_harmonize_poc.py
"""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from snapstudio import config  # noqa: E402
from snapstudio.reshape import _dominant_hue, _face_region, _on_white  # noqa: E402
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL  # noqa: E402

OUT = ROOT / "examples" / "outputs" / "watch_worn"
GEN = OUT / "real_s0.45_seed42.png"
REAL = ROOT / "examples" / "products" / "user" / "watch_front_onwhite.png"


def _dial_cr(rgb, hue):
    res = _face_region(rgb, hue)
    if res is None:
        return None
    _, cnt = res
    (cx, cy), (MA, ma), _ = cv2.fitEllipse(cnt)
    return cx, cy, (MA + ma) / 4.0


def _forearm_angle(gen_rgb):
    """用膚色(YCrCb)偵測前臂，PCA 求主軸角度(度，0=水平)；失敗回 None。"""
    ycrcb = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 135, 80), (255, 180, 130))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.02 * gen_rgb.shape[0] * gen_rgb.shape[1]:
        return None
    ys, xs = np.where(cv2.drawContours(np.zeros(skin.shape, np.uint8), [c], -1, 255, -1) > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, eig = cv2.PCACompute(pts, mean=None)
    vx, vy = eig[0]
    if vy > 0:  # 讓主軸指向上方(朝手部，影像上方 y 較小)
        vx, vy = -vx, -vy
    return float(np.degrees(np.arctan2(vy, vx)))  # 影像座標(y下)，up≈-90


def build_composite_and_map(gen_img, real_img, rot_deg=0.0):
    """回傳 (合成底圖 PIL, 變更量 map PIL L)。真實錶頭旋轉 rot_deg 後貼到生成錶位置。"""
    gen = np.array(gen_img.convert("RGB"))
    real = np.array(real_img.convert("RGB"))
    hue = _dominant_hue(real)
    gcr, rcr = _dial_cr(gen, hue), _dial_cr(real, hue)
    if gcr is None or rcr is None:
        raise RuntimeError("錶盤偵測失敗")
    gcx, gcy, grad = gcr
    rcx, rcy, rrad = rcr
    H, W = gen.shape[:2]

    # 真實錶頭：以真實錶盤為中心，裁 1.55*半徑見方（含錶殼），連 alpha
    real_rgba = np.array(Image.open(REAL.parent / "watch_front.png").convert("RGBA"))
    head = int(rrad * 1.55)
    y0, y1 = max(0, int(rcy - head)), min(real_rgba.shape[0], int(rcy + head))
    x0, x1 = max(0, int(rcx - head)), min(real_rgba.shape[1], int(rcx + head))
    crop = real_rgba[y0:y1, x0:x1]
    # 縮放：真實錶盤半徑 → 生成錶盤半徑
    scale = grad / max(rrad, 1.0)
    nh, nw = int(crop.shape[0] * scale), int(crop.shape[1] * scale)
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    # 旋轉錶頭對齊前臂方向（PIL 逆時針為正）
    if abs(rot_deg) > 0.5:
        crop = np.array(Image.fromarray(crop).rotate(rot_deg, expand=True,
                                                      resample=Image.BICUBIC))
    nh, nw = crop.shape[0], crop.shape[1]

    # 貼到生成錶位置（錶頭中心對齊生成錶盤中心）
    px, py = int(gcx - nw / 2), int(gcy - nh / 2)
    canvas = Image.fromarray(gen).convert("RGBA")
    head_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    head_layer.paste(Image.fromarray(crop), (px, py), Image.fromarray(crop))
    comp = Image.alpha_composite(canvas, head_layer).convert("RGB")

    # 變更量 map（差分擴散語意：map=1 凍結保真、map=0 自由重繪）：
    # 錶頭=1凍結(保真實TISSOT)；環繞區(含錶帶)=自由重繪→讓 diffusion 重畫錶帶對齊轉過的錶頭；
    # 外圍場景=0.9近凍結(保留手腕/背景)。
    # 錶頭凍結保真；錶頭周圍(錶帶區)自由重繪→錶帶跟著旋轉的錶頭對齊；外圍手腕/場景凍結
    alpha = np.array(head_layer.split()[-1])
    headm = (alpha > 128).astype(np.uint8) * 255
    core = cv2.erode(headm, np.ones((7, 7), np.uint8))
    surround = np.zeros((H, W), np.uint8)
    cv2.circle(surround, (int(gcx), int(gcy)), int(grad * 2.7), 255, -1)  # 錶帶區(加寬)
    # 抹掉原本直立錶帶(填膚色)→逼模型從旋轉後的錶耳重畫對齊的錶帶，而非保留舊的
    skinm = cv2.inRange(cv2.cvtColor(gen, cv2.COLOR_RGB2YCrCb), (0, 135, 80), (255, 180, 130))
    sk_px = gen[skinm > 0]
    if len(sk_px) > 100:
        med = np.median(sk_px, axis=0).astype(np.uint8)
        erase = cv2.bitwise_and(surround, cv2.bitwise_not(cv2.dilate(headm, np.ones((9, 9), np.uint8))))
        ca = np.array(comp)
        ca[erase > 0] = med
        ca = cv2.GaussianBlur(ca, (0, 0), 2)
        keep = (erase == 0)
        ca[keep] = np.array(comp)[keep]
        comp = Image.fromarray(ca)
    m = np.full((H, W), 1.0, np.float32)       # 外圍手腕/場景：凍結(保住手腕角度)
    m[surround > 0] = 0.0                       # 錶帶區：全自由→重畫對齊旋轉後的錶頭
    m[core > 0] = 1.0                           # 錶頭：凍結保真
    m = cv2.GaussianBlur(m, (0, 0), 4)
    m[core > 0] = 1.0
    return comp, Image.fromarray((m * 255).astype(np.uint8), "L")


def main():
    import os
    gen_img = Image.open(GEN)
    fa = _forearm_angle(np.array(gen_img.convert("RGB")))
    print(f"前臂角度(影像座標,up≈-90) = {fa}", flush=True)
    target = (fa + 90.0) if fa is not None else 0.0  # 把錶頭up(-90)轉到前臂方向

    if os.environ.get("COMPOSITE_ONLY"):
        # 掃一排明確旋轉角給人挑（PIL 逆時針為正；正角=錶頭逆時針轉，頂端朝左上）
        for rot in (-30, -20, -10, 10, 20, 30):
            comp, _ = build_composite_and_map(gen_img, Image.open(REAL), rot_deg=float(rot))
            comp.save(OUT / f"hp_comp_r{rot:+d}.png")
            print(f"存 hp_comp_r{rot:+d}.png", flush=True)
        print("COMPOSITE_ONLY_DONE", flush=True)
        return

    rot = float(os.environ.get("ROT", "10"))  # 逆時針(rotN 方向)，可用 ROT 環境變數調
    comp, cmap = build_composite_and_map(gen_img, Image.open(REAL), rot_deg=rot)
    comp.save(OUT / "hp_composite.png")
    cmap.save(OUT / "hp_map.png")
    print(f"composite + map 完成 (rot={rot:.0f})", flush=True)

    spec = importlib.util.spec_from_file_location(
        "diff", str(ROOT / "snapstudio" / "pipelines" / "diff_img2img.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    DiffPipe = next(getattr(mod, n) for n in dir(mod)
                    if n.endswith("Pipeline") and "Differential" in n)
    vae = AutoencoderKL.from_pretrained(config.SDXL_VAE_ID, torch_dtype=torch.float16)
    base = StableDiffusionXLImg2ImgPipeline.from_single_file(
        str(config.REALVISXL_PATH), torch_dtype=torch.float16, vae=vae)
    pipe = DiffPipe(**base.components).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    map_t = torchvision.transforms.ToTensor()(cmap).half().to("cuda")  # [1,H,W]，勿多加 batch 維
    orig_t = pipe.image_processor.preprocess(comp)  # original_image 需 preprocess 成像素張量
    prompt = ("close-up of a luxury wristwatch worn on a man's wrist, the stainless steel "
              "bracelet wrapping around the wrist and aligned with the watch case, natural "
              "skin, soft studio light, cohesive lighting and contact shadow, photorealistic, "
              "high detail")
    neg = "deformed, blurry, lowres, distorted watch, melted, extra dial, text artifacts"
    for strg in (0.85, 0.95):
        g = torch.Generator("cuda").manual_seed(7)
        img = pipe(prompt=prompt, negative_prompt=neg, image=orig_t, original_image=orig_t,
                   map=map_t, num_inference_steps=30, strength=strg, guidance_scale=6.0,
                   generator=g).images[0]
        img.save(OUT / f"hp_harmonized_s{strg}.png")
        print(f"done strength={strg}", flush=True)
    print("HARMONIZE_POC_DONE", flush=True)


if __name__ == "__main__":
    main()
