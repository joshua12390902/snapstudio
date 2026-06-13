"""Ultracode 全展示品生成（serial GPU）：watch 走鎖定，其餘 auto。
輸出 examples/outputs/review2/<name>_p{0,1}.png + meta.json（含每階段計時）。"""
import sys, os
sys.path.insert(0, '/workspace/Deep_Generative_Model/HW7_snapstudio')
import json, time, traceback
from PIL import Image
from snapstudio.pipeline import SnapStudio

OUT = "examples/outputs/review2"
os.makedirs(OUT, exist_ok=True)

# (名稱, 原圖, 需求, mode)
JOBS = [
    ("watch",        "examples/products/raw/user/watch_front.jpg", "沉著冷靜 高級腕錶電商大圖", "locked"),
    ("perfume",      "examples/products/user/perfume_front.png",   "質感清心 賣給輕熟女",        "auto"),
    ("energy_drink", "examples/products/energy_drink.png",         "清涼能量飲料廣告",          "auto"),
    ("sneaker",      "examples/products/sneaker.png",              "潮流球鞋電商廣告",          "auto"),
    ("wallet",       "examples/products/wallet.png",               "皮夾質感商品照",            "auto"),
    ("lipstick",     "examples/products/lipstick.png",             "精品唇膏質感大圖",          "auto"),
    ("controller",   "examples/products/controller.png",           "遊戲手把帥氣手持",          "auto"),
    ("earbuds",      "examples/products/earbuds.png",              "無線耳機質感廣告",          "auto"),
]

s = SnapStudio()
meta = []
for name, path, brief, mode in JOBS:
    try:
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        res = s.process(img, user_brief=brief, n_plans=2, mode=mode)
        for i, sh in enumerate(res.shots):
            sh.save(f"{OUT}/{name}_p{i}.png")
        rec = {"name": name, "mode": res.mode, "product_class": res.product_class,
               "name_guess": res.card.name_guess, "worn_framing": getattr(res.card, "worn_framing", ""),
               "sec": round(time.time() - t0), "timings": res.timings}
        meta.append(rec)
        print(f"DONE {name}: {res.mode} {rec['sec']}s timings={res.timings}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {e}", flush=True)
        traceback.print_exc()
        meta.append({"name": name, "error": str(e)})
    json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=2)
print("BATCH_V2_DONE", flush=True)
