"""生成 agent 找來的新來源商品（examples/products/sourced/），輸出 review3。auto 路由。"""
import sys, os
sys.path.insert(0, '/workspace/Deep_Generative_Model/HW7_snapstudio')
import json, time, traceback
from PIL import Image
from snapstudio.pipeline import SnapStudio

SRC = "examples/products/sourced"
OUT = "examples/outputs/review3"
os.makedirs(OUT, exist_ok=True)
BRIEF = {
    "controller": "遊戲手把電商商品圖",
    "headphones": "耳罩式耳機質感商品圖",
    "sunglasses": "墨鏡精品商品圖",
    "handbag":    "真皮手提包精品商品圖",
    "watch2":     "高級腕錶電商大圖",
}
s = SnapStudio()
meta = json.load(open(f"{OUT}/meta.json")) if os.path.exists(f"{OUT}/meta.json") else []
for name in sys.argv[1:]:
    try:
        img = Image.open(f"{SRC}/{name}.jpg").convert("RGB")
        t0 = time.time()
        res = s.process(img, user_brief=BRIEF.get(name, "精品商品圖"), n_plans=2, mode="auto")
        for i, sh in enumerate(res.shots):
            sh.save(f"{OUT}/{name}_p{i}.png")
        meta.append({"name": name, "mode": res.mode, "product_class": res.product_class,
                     "best_shot": getattr(res.card, "best_shot", ""), "name_guess": res.card.name_guess,
                     "sec": round(time.time() - t0)})
        print(f"DONE {name}: {res.mode} best_shot={getattr(res.card,'best_shot','')} {round(time.time()-t0)}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {e}", flush=True); traceback.print_exc()
    json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=2)
print("GEN_SOURCED_DONE", flush=True)
