"""補齊剩餘產品（新 32b-ctx8k 管線）。用 sys.argv 指定要跑哪些，分批避免背景逾時。"""
import sys, os
sys.path.insert(0, '/workspace/Deep_Generative_Model/HW7_snapstudio')
import json, time, traceback
from PIL import Image
from snapstudio.pipeline import SnapStudio

OUT = "examples/outputs/review2"
os.makedirs(OUT, exist_ok=True)
ALL = {
    "watch":        ("examples/products/raw/user/watch_front.jpg", "沉著冷靜 高級腕錶電商大圖", "locked"),
    "perfume":      ("examples/products/user/perfume_front.png",   "質感清心 賣給輕熟女",       "auto"),
    "energy_drink": ("examples/products/energy_drink.png",         "清涼能量飲料廣告",          "auto"),
    "sneaker":    ("examples/products/sneaker.png",    "潮流球鞋電商廣告", "auto"),
    "wallet":     ("examples/products/wallet.png",     "皮夾質感商品照",   "auto"),
    "lipstick":   ("examples/products/lipstick.png",   "精品唇膏質感大圖", "auto"),
    "controller": ("examples/products/controller.png", "遊戲手把帥氣手持", "auto"),
    "earbuds":    ("examples/products/earbuds.png",    "無線耳機質感廣告", "auto"),
}
want = sys.argv[1:] or list(ALL)
s = SnapStudio()
meta_path = f"{OUT}/meta_rest.json"
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else []
for name in want:
    path, brief, mode = ALL[name]
    try:
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        res = s.process(img, user_brief=brief, n_plans=2, mode=mode)
        for i, sh in enumerate(res.shots):
            sh.save(f"{OUT}/{name}_p{i}.png")
        rec = {"name": name, "mode": res.mode, "product_class": res.product_class,
               "name_guess": res.card.name_guess, "sec": round(time.time() - t0)}
        meta.append(rec)
        print(f"DONE {name}: {res.mode} {rec['sec']}s class={res.product_class}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {e}", flush=True); traceback.print_exc()
    json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=2)
print("BATCH_REST_DONE", flush=True)
