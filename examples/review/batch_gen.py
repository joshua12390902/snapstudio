"""多真實產品批次測試：每個產品跑一遍 pipeline（auto 路由），存成品供嚴格審查。
GPU 序列、單張安全。輸出 examples/outputs/review/<name>_shot.png + meta.json。"""
import sys, os
sys.path.insert(0, '/workspace/Deep_Generative_Model/HW7_snapstudio')
import json, time, traceback
from PIL import Image
from snapstudio.pipeline import SnapStudio

OUT = "examples/outputs/review"
os.makedirs(OUT, exist_ok=True)

# (名稱, 原圖, 口語需求) — 涵蓋 rigid(鎖定) 與 wearable/handheld(重塑)
PRODUCTS = [
    ("perfume",      "examples/products/user/perfume_front.png", "高級香水電商情境，乾淨大片感"),
    ("energy_drink", "examples/products/energy_drink.png",        "清涼能量飲料廣告"),
    ("controller",   "examples/products/controller.png",          "遊戲手把帥氣手持情境"),
    ("sneaker",      "examples/products/sneaker.png",              "潮流球鞋電商廣告"),
    ("earbuds",      "examples/products/earbuds.png",              "無線耳機質感廣告"),
    ("wallet",       "examples/products/wallet.png",               "皮夾質感商品照"),
]

s = SnapStudio()
meta = []
for name, path, brief in PRODUCTS:
    try:
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        res = s.process(img, user_brief=brief, n_plans=1, mode="auto")
        res.shots[0].save(f"{OUT}/{name}_shot.png")
        rec = {"name": name, "mode": res.mode, "product_class": res.product_class,
               "worn_framing": getattr(res.card, "worn_framing", ""),
               "name_guess": res.card.name_guess, "sec": round(time.time() - t0)}
        meta.append(rec)
        print(f"DONE {name}: mode={res.mode} class={res.product_class} "
              f"name={res.card.name_guess!r} {rec['sec']}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {e}", flush=True)
        traceback.print_exc()
        meta.append({"name": name, "error": str(e)})
json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=2)
print("BATCH_GEN_DONE", flush=True)
