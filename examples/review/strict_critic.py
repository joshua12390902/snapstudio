"""嚴格審查員：對 batch_gen 產出的每張成品，用 VLM 當「極度嚴格的廣告攝影總監」
逐張挑瑕疵、評分、判 pass/fail。一次載 VLM 跑完全部（少 VRAM 來回）。
輸出 examples/outputs/review/critique.json + 終端報告。"""
import sys, os
sys.path.insert(0, '/workspace/Deep_Generative_Model/HW7_snapstudio')
import json
from PIL import Image
from snapstudio import config
from snapstudio.llm import LLMClient, _image_to_data_url, _extract_json

OUT = "examples/outputs/review"
meta = json.load(open(f"{OUT}/meta.json"))

STRICT_PROMPT = (
    "你是頂尖品牌的廣告攝影總監，以**極度嚴格**著稱。這是一張「{desc}」的 AI 生成商品"
    "廣告成品（{mode_hint}）。請用挑剔的眼光找出**所有**破綻，標準是『能不能直接登上"
    "電商首頁大圖』，不行就嚴懲。\n"
    "檢查：①產品本體是否變形/比例怪/失真 ②是否有雙重或衝突陰影、浮空不接地 "
    "③是否出現重複/第二個產品、拼接、平鋪 ④穿戴類是否真的自然戴/握在正確身體部位、"
    "大小合理 ⑤亂碼文字/假 logo ⑥光影與場景是否一致 ⑦整體是否廉價/詭異。\n"
    '只輸出 JSON：{{"score": 1-10, "pass": true/false, '
    '"verdict": "一句總評（繁體中文）", "flaws": ["具體瑕疵，繁體中文，從最嚴重排"], '
    '"fix_hint": "若要改善，最該動哪裡（繁體中文）"}}\n'
    "score>=8 且無明顯破綻才 pass=true。沒瑕疵就 flaws=[]。務必繁體中文、不可簡體。"
)


def critique(llm, img, desc, mode):
    mode_hint = "穿戴/手持重塑" if mode == "reshape" else "剛性產品擺台"
    prompt = STRICT_PROMPT.format(desc=desc, mode_hint=mode_hint)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_to_data_url(img)}}]}]
    for _ in range(3):
        try:
            raw = llm._chat_once(llm._ollama, config.OLLAMA_VISION_MODEL, messages, 0.2)
            d = _extract_json(raw)
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return None


llm = LLMClient()
llm._ensure_vision()
report = []
for rec in meta:
    name = rec["name"]
    p = f"{OUT}/{name}_shot.png"
    if "error" in rec or not os.path.exists(p):
        print(f"[{name}] 無成品（{rec.get('error','缺圖')}）", flush=True)
        continue
    desc = rec.get("name_guess") or name
    c = critique(llm, Image.open(p).convert("RGB"), desc, rec.get("mode", "locked"))
    rec["critique"] = c
    report.append(rec)
    if c:
        mark = "✅PASS" if c.get("pass") else "❌FAIL"
        print(f"[{name}] {mark} score={c.get('score')} mode={rec.get('mode')} | "
              f"{c.get('verdict')}", flush=True)
        for f in (c.get("flaws") or []):
            print(f"     - {f}", flush=True)
        if c.get("fix_hint"):
            print(f"     → 修法：{c.get('fix_hint')}", flush=True)
    else:
        print(f"[{name}] 審查失敗（VLM 無回應）", flush=True)
json.dump(report, open(f"{OUT}/critique.json", "w"), ensure_ascii=False, indent=2)
llm.release_models()
print("CRITIC_DONE", flush=True)
