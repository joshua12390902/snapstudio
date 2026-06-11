"""混合法 POC：重塑（戴手腕）已生成 → 偵測生成圖的錶盤 → 把真實錶面
homography 變形貼回 → 羽化融合，讓 logo/刻度=真實（解決亂碼字）。

通用思路：穿戴/手持重塑後，對「平面細節區」（錶盤/標籤/logo 面）用真實像素
合成回去，補回 IP-Adapter 重塑必然丟失的小細節。本檔先以手錶錶盤示範。

用法：PYTHONPATH=. python examples/hybrid_dial.py <生成worn圖> <真實watch_onwhite>
輸出：examples/wornmode/hybrid_<...>.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "wornmode"


def dial_region(rgb: np.ndarray):
    """偵測藍色錶盤 → 回 (填滿凸包的 dial 遮罩, 四角點 TL,TR,BR,BL)；失敗回 None。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (90, 55, 25), (135, 255, 235))  # 藍盤
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    h, w = rgb.shape[:2]
    if cv2.contourArea(c) < 0.008 * h * w:
        return None
    hull = cv2.convexHull(c)
    fill = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(fill, hull, 255)          # 填滿錶盤（含刻度/指針/文字）
    box = cv2.boxPoints(cv2.minAreaRect(c))      # 旋轉外接矩形四角
    return fill, _order_quad(box)


def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    ys = pts[np.argsort(pts[:, 1])]
    top = ys[:2][np.argsort(ys[:2, 0])]   # TL, TR
    bot = ys[2:][np.argsort(ys[2:, 0])]   # BL, BR
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)  # TL,TR,BR,BL


def hybrid(gen_path: str, real_path: str) -> Path:
    gen = Image.open(gen_path).convert("RGB")
    real = Image.open(real_path).convert("RGB")
    gen_arr, real_arr = np.array(gen), np.array(real)

    g = dial_region(gen_arr)
    r = dial_region(real_arr)
    if g is None or r is None:
        raise RuntimeError(f"錶盤偵測失敗 gen={g is not None} real={r is not None}")
    gen_fill, gen_quad = g
    real_fill, real_quad = r

    # 真實錶盤 → 生成錶盤 的透視變形
    H = cv2.getPerspectiveTransform(real_quad, gen_quad)
    Hh, Hw = gen_arr.shape[:2]
    warped = cv2.warpPerspective(real_arr, H, (Hw, Hh))
    warped_mask = cv2.warpPerspective(real_fill, H, (Hw, Hh))
    # 只在「生成錶盤 ∩ 變形後真實錶盤」區域合成，避免溢出到錶框/手腕
    region = cv2.bitwise_and(gen_fill, warped_mask)
    region = cv2.erode(region, np.ones((3, 3), np.uint8))  # 內縮避免吃到錶框

    # 羽化邊界做無縫融合
    alpha = Image.fromarray(region).filter(ImageFilter.GaussianBlur(2.5))
    out = Image.composite(Image.fromarray(warped), gen, alpha)
    name = f"hybrid_{Path(gen_path).stem}.png"
    out.save(OUT / name)
    # 同時存偵測可視化（debug）
    dbg = gen_arr.copy()
    cv2.polylines(dbg, [gen_quad.astype(int)], True, (255, 0, 0), 3)
    Image.fromarray(dbg).save(OUT / f"dbg_{Path(gen_path).stem}.png")
    return OUT / name


if __name__ == "__main__":
    gp = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "real_s0.45_seed42.png")
    rp = sys.argv[2] if len(sys.argv) > 2 else str(
        ROOT / "examples" / "products" / "user" / "watch_front_onwhite.png")
    p = hybrid(gp, rp)
    print(f"HYBRID_DONE {p}", flush=True)
