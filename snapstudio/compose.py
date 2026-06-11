"""產品擺放與接地合成：把去背產品放到畫布的指定位置/角度/大小，
生成接地陰影，並輸出 inpainting 所需的 init 圖與遮罩。

設計目的（解決「貼紙感」）：
- 使用者可控的 Placement（scale/x/y/rotation）→ 主角能移動、旋轉、縮放
- contact shadow：產品底部的柔和投影，給「站在平面上」的接地感
- inpaint 遮罩：鎖住產品像素，讓擴散模型在產品「周圍」生成場景與表面，
  邊界羽化避免硬邊；陰影區也開放 inpaint，讓模型把影子畫進場景光線
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class Placement:
    """主角在畫布上的擺放參數（皆為相對比例，與畫布尺寸無關）。

    預設經審查員優化：產品偏小（0.40）且偏下（cy 0.68）→ 上方留給背景/海平線、
    下方留給檯面，整合與「背景達成度」最佳；過大或置中會把背景擠掉。
    """
    scale: float = 0.40      # 產品最長邊佔畫布最長邊的比例
    cx: float = 0.5          # 產品中心 x（0~1，0.5=置中）
    cy: float = 0.68         # 產品中心 y（0~1，>0.5 偏下，給上方背景與下方平面）
    rotation: float = 0.0    # 旋轉角度（度，正為逆時針，PIL 慣例）

    def clamped(self) -> "Placement":
        return Placement(
            scale=float(min(max(self.scale, 0.15), 0.95)),
            cx=float(min(max(self.cx, 0.0), 1.0)),
            cy=float(min(max(self.cy, 0.0), 1.0)),
            rotation=float(self.rotation),
        )


def _fit_product(rgba: Image.Image, canvas: tuple[int, int], p: Placement
                 ) -> tuple[Image.Image, tuple[int, int]]:
    """把產品裁去透明邊、旋轉、依 scale 縮放，回傳 (RGBA 產品, 貼上左上角座標)。"""
    rgba = rgba.convert("RGBA")
    bbox = rgba.split()[-1].getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    if abs(p.rotation) > 0.01:
        rgba = rgba.rotate(p.rotation, expand=True, resample=Image.BICUBIC)

    cw, ch = canvas
    target = p.scale * max(cw, ch)
    ratio = target / max(rgba.width, rgba.height)
    new_w = max(1, round(rgba.width * ratio))
    new_h = max(1, round(rgba.height * ratio))
    fg = rgba.resize((new_w, new_h), Image.LANCZOS)

    x = round(p.cx * cw - new_w / 2)
    y = round(p.cy * ch - new_h / 2)
    return fg, (x, y)


def _contact_shadow(canvas: tuple[int, int], fg: Image.Image, pos: tuple[int, int],
                    *, blur: float = 0.045, opacity: float = 0.55,
                    squash: float = 0.18) -> Image.Image:
    """產品底部的柔和接地陰影（L 模式，0=無影 255=最濃）。

    作法：取產品 alpha → 垂直壓扁成橢圓投影 → 貼在產品底緣 → 高斯模糊。
    blur/squash 以畫布最長邊為比例尺，產品越大影子越大、隨之模糊。
    """
    cw, ch = canvas
    longest = max(cw, ch)
    alpha = np.asarray(fg.split()[-1], dtype=np.float32) / 255.0
    fw, fh = fg.width, fg.height

    # 取產品最底一列的水平輪廓寬度，作為影子橢圓的長軸
    cols = alpha.max(axis=0)  # 每欄是否有產品
    xs = np.where(cols > 0.1)[0]
    if len(xs) == 0:
        return Image.new("L", canvas, 0)
    left, right = xs.min(), xs.max()
    shadow_w = right - left
    shadow_h = max(4, int(fh * squash))

    shadow = Image.new("L", canvas, 0)
    ellipse = Image.new("L", (max(1, shadow_w), shadow_h), 0)
    ea = np.zeros((shadow_h, max(1, shadow_w)), dtype=np.float32)
    yy, xx = np.ogrid[:shadow_h, :max(1, shadow_w)]
    cxp, cyp = max(1, shadow_w) / 2, shadow_h / 2
    rx, ry = max(1, shadow_w) / 2, shadow_h / 2
    mask = ((xx - cxp) / rx) ** 2 + ((yy - cyp) / ry) ** 2 <= 1.0
    ea[mask] = 255 * opacity
    ellipse = Image.fromarray(ea.astype(np.uint8))

    # 貼在產品底緣（略微上移讓影子與產品接觸）
    px, py = pos
    bottom_y = py + fh - shadow_h // 2
    shadow.paste(ellipse, (px + (left), bottom_y))
    return shadow.filter(ImageFilter.GaussianBlur(blur * longest))


def build_scene_inputs(
    rgba: Image.Image,
    canvas: tuple[int, int],
    placement: Placement,
    *,
    feather: float = 0.004,
    seed: int = 0,
) -> dict:
    """產出 inpaint 三件套與預覽。

    回傳：
        init         RGB：高頻雜訊底 + 產品，餵 SDXL inpaint 的 image
        mask         L  ：白=待生成（產品以外），黑=保留（產品本體），邊界羽化
        product      RGBA：擺好位置的產品（含 alpha，供最後貼回鎖死像素）
        product_gray RGB：產品依**相同擺放**貼在 127 灰底（給 IC-Light 對齊用，
                          避免前景置中而與場景中的產品位置錯開造成「重疊」）
        preview      RGB：中性灰底上的擺放預覽（給 UI 顯示，非餵模型）

    feather 預設小（0.004）：羽化太大會讓 inpaint 吃進產品邊緣、產生模糊光暈。

    關鍵（實測）：init 的待生成區必須是**高頻雜訊**而非純色。純灰/純白會把
    inpaint 輸出錨定成同一純色（denoiser 沒有可雕刻的內容）；雜訊則讓模型把
    它去噪成場景細節，接地陰影與反光由模型在同一次生成中自然長出。
    """
    p = placement.clamped()
    cw, ch = canvas
    fg, pos = _fit_product(rgba, canvas, p)
    px, py = pos

    # 產品 alpha 貼到全畫布尺寸的圖層（後續所有遮罩都基於它）
    prod_layer = Image.new("RGBA", canvas, (0, 0, 0, 0))
    prod_layer.paste(fg, (px, py), fg)
    prod_alpha = prod_layer.split()[-1]

    # init：種子化高頻雜訊底 → 貼上產品（產品區之後被 mask 保護不變）
    rng = np.random.RandomState(seed)
    noise = (rng.rand(ch, cw, 3) * 255).astype(np.uint8)
    init = Image.fromarray(noise, "RGB")
    init.paste(fg, (px, py), fg)

    # 產品依相同擺放貼到 127 灰底（IC-Light 前景格式，位置與場景一致）
    product_gray = Image.new("RGB", canvas, (127, 127, 127))
    product_gray.paste(fg, (px, py), fg)

    # 預覽圖另用中性灰底（給人看「主角擺哪」，不餵模型）
    preview = Image.new("RGB", canvas, (200, 200, 200))
    preview.paste(fg, (px, py), fg)

    # mask：白底（全要生成）→ 把產品塗黑（保留），再羽化邊界
    mask = Image.new("L", canvas, 255)
    mask.paste(Image.new("L", prod_alpha.size, 0), (0, 0), prod_alpha)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather * max(cw, ch)))

    return {"init": init, "mask": mask, "product": prod_layer,
            "product_gray": product_gray, "preview": preview}


def paste_back(generated: Image.Image, product: Image.Image,
               *, shadow: bool = True, light_direction: str = "top",
               shadow_opacity: float = 0.55) -> Image.Image:
    """把鎖死的產品像素貼回生成結果，並在底部加方向性接地投影（解決浮貼感）。

    （電商鐵則：商品本體不可被模型幻想改變；只有周圍場景是生成的。）
    shadow：合成兩層——緊貼底部的「接觸核」（暗、銳）＋依光向斜投的「投射影」
    （長、柔），讓立式產品真正「站」在地面而非浮貼。
    light_direction：光源方向，投影往反方向延伸（left→影子偏右，依此類推）。
    """
    out = generated.convert("RGB").copy()
    # 抗白暈：把 alpha 邊緣內縮 1px，去掉 rembg 殘留的半透明亮邊
    product = product.convert("RGBA")
    r, g, b, a = product.split()
    a = a.filter(ImageFilter.MinFilter(3))
    product = Image.merge("RGBA", (r, g, b, a))
    if shadow:
        alpha = a
        bbox = alpha.getbbox()
        if bbox:
            cw, ch = out.size
            x0, y0, x1, y1 = bbox
            pw, ph = x1 - x0, y1 - y0
            base_cx, base_y = (x0 + x1) // 2, y1
            dx = {"left": 1, "right": -1}.get(light_direction, 0.4)
            dark = Image.new("RGB", (cw, ch), (0, 0, 0))
            # 第1層 投射影：中長橢圓往光反向（柔、淡）
            cast = Image.new("L", (cw, ch), 0)
            cl = int(pw * 1.25); co = int(dx * pw * 0.4)
            ImageDraw.Draw(cast).ellipse(
                [base_cx - cl // 2 + co, base_y - int(ph * 0.04),
                 base_cx + cl // 2 + co, base_y + int(ph * 0.13)],
                fill=int(255 * shadow_opacity * 0.55))
            cast = cast.filter(ImageFilter.GaussianBlur(max(6, ph * 0.05)))
            out = Image.composite(dark, out, cast)
            # 第2層 接觸核：緊貼底部的深橢圓（接地主力）
            core = Image.new("L", (cw, ch), 0)
            cw2 = int(pw * 0.88)
            ImageDraw.Draw(core).ellipse(
                [base_cx - cw2 // 2, base_y - int(ph * 0.03),
                 base_cx + cw2 // 2, base_y + int(ph * 0.06)],
                fill=int(255 * min(0.92, shadow_opacity * 1.4)))
            core = core.filter(ImageFilter.GaussianBlur(max(3, ph * 0.03)))
            out = Image.composite(dark, out, core)
            # 第3層 接觸線(ambient occlusion)：產品正下方最暗、最緊
            ao = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(ao).ellipse(
                [base_cx - int(pw * 0.44), base_y - int(ph * 0.02),
                 base_cx + int(pw * 0.44), base_y + int(ph * 0.035)],
                fill=int(255 * min(0.98, shadow_opacity * 1.7)))
            ao = ao.filter(ImageFilter.GaussianBlur(max(2, ph * 0.013)))
            out = Image.composite(dark, out, ao)
    out.paste(product, (0, 0), product)
    return out
