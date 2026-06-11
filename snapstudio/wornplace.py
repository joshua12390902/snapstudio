"""AnyDoor 穿戴/手持擺放：跨 env（py3.8/torch2.0）以 subprocess 呼叫 AnyDoor，
把產品以自然朝向/光照/環繞「搬」到生成的身體部位場景上。

為何跨 env：AnyDoor 自帶 ldm/cldm 釘死 torch2.0/pl1.5，與主 pipeline 的 diffusers
0.39 不相容；用 subprocess 把兩個環境隔開，主 pipeline 寫入產品/場景/遮罩、AnyDoor
env 跑推論、回傳成品 PNG。配 reshape.composite_real_face 把真實平面細節（錶盤/標籤）
合成回去 → 自然戴上身 ＋ 真實 logo 兼顧。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ANYDOOR_PY = "/miniconda/envs/anydoor/bin/python"
ANYDOOR_DIR = "/workspace/AnyDoor"
ANYDOOR_CLI = "anydoor_cli.py"


def available() -> bool:
    """AnyDoor env 與權重是否就緒（沒有就讓 pipeline 退回 IP-Adapter 重塑）。"""
    return (Path(ANYDOOR_PY).exists()
            and (Path(ANYDOOR_DIR) / "path" / "epoch=1-step=8687.ckpt").exists()
            and (Path(ANYDOOR_DIR) / "path" / "dinov2_vitg14_pretrain.pth").exists())


def placement_mask(size, cx=0.6, cy=0.42, rx=0.14, ry=0.22) -> Image.Image:
    """場景上「產品要放哪」的橢圓遮罩（白=放這）。固定中央，偵測失敗時的退路。"""
    w, h = size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse(
        [int((cx - rx) * w), int((cy - ry) * h),
         int((cx + rx) * w), int((cy + ry) * h)], fill=255)
    return m


def mask_from_box(size, box) -> Image.Image | None:
    """VLM 回的 0-1000 正規化框 [x0,y0,x1,y1] → 橢圓擺放遮罩。框不合理回 None。"""
    if not box or len(box) != 4:
        return None
    x0, y0, x1, y1 = [c / 1000.0 for c in box]
    if not (0 <= x0 < x1 <= 1.001 and 0 <= y0 < y1 <= 1.001):
        return None
    if (x1 - x0) > 0.92 or (y1 - y0) > 0.92:  # 佔滿整張＝VLM 沒抓準，棄用
        return None
    w, h = size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).ellipse([int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)], fill=255)
    return m


def body_part_mask(scene_img: Image.Image, product_class: str = "wearable",
                   scale: float = 0.5) -> Image.Image:
    """自動偵測場景裡的身體部位（膚色），把擺放遮罩對準它、依手臂寬度定大小、
    沿手臂方向定位（手錶/手環→腕部偏手端；其餘→部位中心）。偵測失敗退回固定中央。"""
    import cv2
    import numpy as np
    arr = np.array(scene_img.convert("RGB"))
    h, w = arr.shape[:2]
    ycc = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycc, (0, 135, 80), (255, 180, 130))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.03 * h * w]
    if not cnts:
        return placement_mask((w, h))
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)  # 手臂的旋轉外接矩形
    arm_w = max(8.0, min(rw, rh))                  # 手臂寬度（短邊）
    long_v = (np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang)))
    if rh > rw:  # 長軸是另一邊
        long_v = (-np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)))
    # 手錶/手環：沿手臂長軸往「靠手端（影像上方，y 小）」移一點到腕部
    if product_class == "wearable":
        step = max(rw, rh) * 0.18
        if long_v[1] > 0:  # 讓位移指向上方(手端)
            long_v = (-long_v[0], -long_v[1])
        cx += long_v[0] * step
        cy += long_v[1] * step
    # 產品案體比例由 scale 控制（VLM 判太大時 pipeline 會以更小的 scale 重跑）
    rad = arm_w * scale
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rad), int(rad * 0.92)),
                ang, 0, 360, 255, -1)
    return Image.fromarray(m, "L")


def place_batch(product_rgba: Image.Image, scenes: list[Image.Image],
                seeds: list[int], masks: list[Image.Image] | None = None) -> list:
    """把 product_rgba 以自然姿態擺到每個 scene 的遮罩處。回傳 list[PIL RGB]（失敗者為 None）。
    一次 subprocess 只載一次 AnyDoor 模型、處理全部 scenes。"""
    tmp = Path(tempfile.mkdtemp(prefix="anydoor_"))
    ref_p = tmp / "ref.png"
    product_rgba.convert("RGBA").save(ref_p)
    jobs, outs = [], []
    for i, scene in enumerate(scenes):
        sp = tmp / f"scene_{i}.png"
        scene.convert("RGB").save(sp)
        mk = masks[i] if masks else placement_mask(scene.size)
        mp = tmp / f"mask_{i}.png"
        mk.save(mp)
        op = tmp / f"out_{i}.png"
        outs.append(op)
        jobs.append({"scene": str(sp), "mask": str(mp),
                     "seed": int(seeds[i]), "out": str(op)})
    (tmp / "spec.json").write_text(json.dumps({"ref": str(ref_p), "jobs": jobs}))

    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = ""  # torch2.0 不認 expandable_segments
    subprocess.run([ANYDOOR_PY, ANYDOOR_CLI, str(tmp / "spec.json")],
                   cwd=ANYDOOR_DIR, env=env, check=True)
    return [Image.open(o).convert("RGB") if o.exists() else None for o in outs]
