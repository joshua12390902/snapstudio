"""去背與前景合成：rembg 偵測 + SAM2 精修邊緣、IC-Light 灰底前景格式。"""
import logging

import numpy as np
from PIL import Image
from rembg import new_session, remove

from . import config

logger = logging.getLogger(__name__)

GRAY = 127  # IC-Light 前景條件的標準灰底值
SAM2_PATH = config.WEIGHTS / "sam" / "sam2_b.pt"


class Matting:
    """rembg 去背 + （選配）SAM2 邊緣精修。

    rembg（birefnet-general）抓出產品大致範圍，再用 SAM2 以 bbox 提示重切，
    得到**零半透明白暈的銳利邊緣**（實測 rembg 邊緣 0.15% 模糊像素 → SAM2 0%），
    直接改善「產品貼回背景的白邊光暈與重疊殘影」。SAM2 不可用時自動退純 rembg。
    """

    def __init__(self, model: str = "birefnet-general", refine_sam: bool = True):
        try:
            self.session = new_session(model)
            self.model = model
        except Exception as exc:
            logger.warning("rembg session「%s」建立失敗（%s），退用 u2net", model, exc)
            self.session = new_session("u2net")
            self.model = "u2net"
        self._sam = None
        self.refine_sam = refine_sam

    def _ensure_sam(self):
        if self._sam is None and self.refine_sam:
            try:
                from ultralytics import SAM
                src = str(SAM2_PATH) if SAM2_PATH.exists() else "sam2_b.pt"
                self._sam = SAM(src)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAM2 載入失敗（%s），改用純 rembg 邊緣", exc)
                self.refine_sam = False
        return self._sam

    def cutout(self, image: Image.Image) -> Image.Image:
        """去背，回傳 RGBA（透明區 RGB 保留原圖色）。SAM2 可用時精修邊緣。"""
        image = image.convert("RGB")
        rgba = remove(image, session=self.session)
        arr = np.array(rgba)
        alpha = arr[..., 3]

        sam = self._ensure_sam()
        if sam is not None:
            ys, xs = np.where(alpha > 128)
            if len(xs) > 0:
                box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                try:
                    res = sam(np.array(image), bboxes=[box], verbose=False)
                    m = res[0].masks.data[0].cpu().numpy()
                    sharp = (m > 0.5).astype(np.uint8) * 255
                    return Image.fromarray(
                        np.dstack([arr[..., :3], sharp]), "RGBA"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SAM2 精修失敗（%s），用 rembg 邊緣", exc)
        return rgba

    @staticmethod
    def compose_gray(rgba: Image.Image, size: tuple) -> Image.Image:
        """IC-Light 前景格式：127 灰底、LANCZOS、保持長寬比置中貼齊（不足處補灰）。

        先在原解析度壓上灰底再縮放，避免對直通 alpha 縮放產生邊緣色暈。
        """
        arr = np.array(rgba.convert("RGBA")).astype(np.float32)
        alpha = arr[..., 3:4] / 255.0
        comp = (arr[..., :3] * alpha + GRAY * (1.0 - alpha)).astype(np.uint8)
        img = Image.fromarray(comp)

        ratio = min(size[0] / img.width, size[1] / img.height)
        new_w = max(1, round(img.width * ratio))
        new_h = max(1, round(img.height * ratio))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", size, (GRAY, GRAY, GRAY))
        canvas.paste(img, ((size[0] - new_w) // 2, (size[1] - new_h) // 2))
        return canvas

    @staticmethod
    def compose_on(
        rgba: Image.Image,
        background: Image.Image,
        scale: float = 0.82,
        y_offset: float = 0.06,
    ) -> Image.Image:
        """把前景貼到背景圖中下位置（fbc 模式用），回傳與背景同尺寸的 RGB。

        scale：前景最長邊相對背景的佔比，以 alpha 邊界框為準（先裁掉透明邊，
        讓比例對準商品本體而非整張圖）；y_offset：自畫面中心再往下偏移的比例
        （相對背景高度）。
        """
        rgba = rgba.convert("RGBA")
        bbox = rgba.split()[-1].getbbox()  # 只看 alpha 通道的非零範圍
        if bbox:
            rgba = rgba.crop(bbox)

        bg = background.convert("RGB").copy()
        ratio = min(bg.width * scale / rgba.width, bg.height * scale / rgba.height)
        new_w = max(1, round(rgba.width * ratio))
        new_h = max(1, round(rgba.height * ratio))
        fg = rgba.resize((new_w, new_h), Image.LANCZOS)

        x = (bg.width - new_w) // 2
        y = (bg.height - new_h) // 2 + round(bg.height * y_offset)
        y = max(0, min(y, bg.height - new_h))  # 偏移後不得超出畫面
        bg.paste(fg, (x, y), fg)
        return bg
