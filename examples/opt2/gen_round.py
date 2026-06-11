"""依配方 JSON 批次生成（多輪優化用）。GPU 序列、安全。

用法：PYTHONPATH=<repo> python examples/opt2/gen_round.py <recipe.json>
recipe.json = {"out_dir": "...", "jobs": [
  {"name","product"(can|wallet),"scale","cx","cy","rotation"(選),"prompt",
   "negative","allow_people"(bool),"harmonize"(bool),"light_direction","seed"}
]}
"""
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from snapstudio.compose import Placement, build_scene_inputs, paste_back  # noqa: E402
from snapstudio.groundgen import SceneInpainter  # noqa: E402

# 真實產品（Wikimedia 自由授權，乾淨幾何/真 logo/真材質，比 AI 假素材真實）
_PROD = ROOT / "examples" / "products"
PRODUCTS = {
    "energy_drink": _PROD / "energy_drink.png", "wallet": _PROD / "wallet.png",
    "earbuds": _PROD / "earbuds.png", "perfume": _PROD / "perfume.png",
    "sneaker": _PROD / "sneaker.png", "controller": _PROD / "controller.png",
    "watch": _PROD / "watch.png", "lipstick": _PROD / "lipstick.png",
    "can": _PROD / "energy_drink.png",  # 別名：can→真 Monster 罐
}


def _product_path(j):
    """job 可給 product_path（相對 repo 根）覆寫；否則用 PRODUCTS 對照表。"""
    if j.get("product_path"):
        return ROOT / j["product_path"]
    return PRODUCTS[j.get("product", "can")]


# TripoSR 換角度（job 給 view_azimuth/view_elevation 時，render 該角度的產品）
_tripo = None
_scene_cache = {}


def _product_at_angle(j):
    """產品 RGBA：指定角度且非近正面 → TripoSR render；否則用原圖(保銳利)。"""
    base = _product_path(j)
    az, el = j.get("view_azimuth"), j.get("view_elevation")
    if az is None and el is None:
        return Image.open(base)
    az, el = float(az or 0), float(el or 10)
    if -15 <= az <= 15 and 0 <= el <= 16:  # 近正面用原圖最銳
        return Image.open(base)
    global _tripo
    try:
        if _tripo is None:
            from snapstudio.tripo3d import TripoProduct
            _tripo = TripoProduct(chunk=4096)
        key = str(base)
        if key not in _scene_cache:
            _scene_cache[key] = _tripo.reconstruct(Image.open(base))
        return _tripo.render_angle(_scene_cache[key], elevation=el, azimuth=az, size=512)
    except Exception as exc:  # noqa: BLE001
        print(f"[tripo fail {j['name']}: {exc}] 用原圖", flush=True)
        return Image.open(base)


def main(recipe_path: str) -> None:
    recipe = json.loads(Path(recipe_path).read_text())
    out_dir = ROOT / recipe["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = recipe["jobs"]

    g = SceneInpainter(accelerate=False)
    relighter = None
    for j in jobs:
        rgba = _product_at_angle(j)
        place = Placement(scale=j["scale"], cx=j["cx"], cy=j["cy"],
                          rotation=j.get("rotation", 0.0))
        parts = build_scene_inputs(rgba, (1024, 1024), place, seed=7)
        ld = j.get("light_direction", "top")
        gen = g.generate(
            parts["init"], parts["mask"], j["prompt"],
            negative_prompt=j.get("negative", ""), steps=j.get("steps", 36),
            guidance_scale=j.get("guidance", 7.0), seed=j.get("seed", 42),
            allow_people=bool(j.get("allow_people", False)),
        )
        shot = paste_back(gen, parts["product"], light_direction=ld)
        if j.get("harmonize"):
            if relighter is None:
                from snapstudio.relight import Relighter
                relighter = Relighter("fbc", quality="fast")
            relit = relighter.relight(
                parts["product_gray"].resize((768, 768)), shot.resize((768, 768)),
                prompt=j["prompt"], width=768, height=768, hires=False, lcm=True,
                seed=j.get("seed", 42),
            ).resize((1024, 1024))
            shot = paste_back(relit, parts["product"], light_direction=ld)
        shot.save(out_dir / f"{j['name']}.png")
        print(f"{j['name']} done", flush=True)
    g.unload()
    del g
    global _tripo
    if _tripo is not None:
        _tripo.unload()
        _tripo = None
    _scene_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("ROUND_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
