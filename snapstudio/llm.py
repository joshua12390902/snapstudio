"""LLM/VLM 客戶端：商品識別、場景企劃、文案生成、口語修改解析。

文字呼叫端點降級鏈：Big Pickle（免 key）→ 本機 Ollama qwen3 → 內建預設模板。
所有結構化呼叫：prompt 強制 JSON → 解析失敗重試 2 次 → pydantic 驗證 →
仍失敗套預設模板。本模組對外永不拋例外（identify_product 失敗回 None）。

實測備註：
- Big Pickle 端點拒絕 ``opencode/big-pickle`` 這個 id，但接受去前綴的
  ``big-pickle``（路由到推理模型，最終答案在 message.content）→ 探測時兩個
  候選名都試。
- qwen3 會輸出 <think>…</think>（截斷時可能未閉合），解析前一律剝除。
"""
from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import urllib.request
from typing import Any, Type, TypeVar

from PIL import Image
from openai import OpenAI
from pydantic import BaseModel

try:  # 套件內使用（snapstudio.llm）
    from . import config
    from .schemas import CopyPack, ProductCard, ScenePlan
except ImportError:  # 直接以腳本執行時的退路
    import config
    from schemas import CopyPack, ProductCard, ScenePlan

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# ---------------------------------------------------------------------------
# 內建預設模板：LLM 全掛時的最後防線（系統永不因 LLM 卡死）
# ---------------------------------------------------------------------------

DEFAULT_PRODUCT_CARD = ProductCard(
    category="一般商品",
    name_guess="待補商品名稱",
    material="未知",
    color="未知",
    condition="二手，狀況良好",
    selling_points=["實品如圖", "快速出貨", "可議價"],
    target_audience="一般消費者",
)

DEFAULT_SCENE_PLANS = [
    ScenePlan(
        plan_name="極簡攝影棚",
        scene_prompt=(
            "minimalist studio product photography, seamless light gray "
            "backdrop, subtle reflection on glossy surface, professional "
            "commercial lighting, high detail"
        ),
        negative_prompt="cluttered, text, watermark,人物, lowres, plastic toy look",
        light_direction="top",
        light_desc="soft diffused studio softbox light from above, gentle falloff",
        mood="乾淨俐落",
        composition_tip="商品置中，上下留白，營造留白質感",
    ),
    ScenePlan(
        plan_name="暖調木質桌面",
        scene_prompt=(
            "warm walnut wood tabletop, blurred cozy interior background, "
            "morning ambiance, shallow depth of field, professional product "
            "photography"
        ),
        negative_prompt="cluttered, plastic, text, watermark, lowres",
        light_direction="left",
        light_desc="warm morning window light from the left, soft long shadows",
        mood="溫暖生活感",
        composition_tip="商品置於右三分線，左側留白給光斑",
    ),
    ScenePlan(
        plan_name="冷調大理石",
        scene_prompt=(
            "white marble surface with subtle gray veins, dark moody "
            "background, elegant premium product photography, crisp focus"
        ),
        negative_prompt="cluttered, text, watermark, warm color cast, lowres",
        light_direction="right",
        light_desc="cool crisp rim light from the right, high contrast shadows",
        mood="高級冷冽",
        composition_tip="低角度拍攝強調份量，背景壓暗突顯主體",
    ),
]


def _default_copy(card: ProductCard) -> CopyPack:
    """以商品卡欄位拼出保底文案。"""
    points = card.selling_points or ["實品如圖", "狀況良好", "快速出貨"]
    return CopyPack(
        shopee_title=f"【{card.material}】{card.name_guess} {card.condition[:6]} {card.category}",
        bullet_points=(points + ["實品拍攝", "下單後 24 小時內出貨"])[:5],
        ig_caption=f"{card.name_guess}，{card.color}配色、{card.material}質感 ✨ 詳情請私訊！",
        hashtags=[f"#{card.category.split('/')[0]}", "#二手好物", "#質感生活"],
    )


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

# 產品類別由 VLM 主判（identify_product 看圖輸出 product_class）。以下關鍵字僅在
# 「VLM 端點掛掉走文字備援」時補判，不覆蓋 VLM 的判斷（與全 app 的 LLM 降級鏈一致）。
_WEARABLE_KW = (
    "watch", "錶", "腕錶", "手錶", "ring", "戒指", "戒環", "戒",
    "bracelet", "bangle", "手環", "手鐲", "手鍊", "necklace", "pendant", "項鍊", "項鏈", "墜",
    "earring", "耳環", "耳釘", "glasses", "sunglasses", "eyewear", "眼鏡", "太陽眼鏡",
    "hat", "cap", "beanie", "帽", "jewelry", "jewellery", "飾品", "穿戴", "wearable",
)
_HANDHELD_KW = (
    "controller", "gamepad", "joystick", "手把", "搖桿", "phone", "smartphone", "手機",
    "mouse", "滑鼠", "pen", "stylus", "筆", "remote", "遙控", "camera", "相機",
)


def fallback_product_class(card) -> str:
    """離線備援分類（VLM 不可用時才呼叫）：先信卡片自帶 product_class（若非預設 rigid），
    再用關鍵字補判。VLM 在線時 pipeline 直接採 card.product_class，不經此函式。"""
    pc = getattr(card, "product_class", "rigid") or "rigid"
    if pc != "rigid":
        return pc
    hay = f"{getattr(card, 'category', '')} {getattr(card, 'name_guess', '')}".lower()
    if any(k in hay for k in _WEARABLE_KW):
        return "wearable"
    if any(k in hay for k in _HANDHELD_KW):
        return "handheld"
    return "rigid"


def _strip_think(text: str) -> str:
    """剝除 qwen3 的 <think> 區塊；未閉合（被截斷）時丟棄其後全部內容。"""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def _extract_json(text: str) -> Any:
    """從 LLM 回覆萃取 JSON：先試全文，再試 code fence，最後試首尾大括號。"""
    text = _strip_think(text)
    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("回覆中找不到可解析的 JSON")


def _image_to_data_url(image: Image.Image, max_side: int = 1024) -> str:
    """PIL 圖 → base64 data URL；先縮到 max_side 以內，控制 payload 大小。"""
    img = image.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


JSON_RULES = (
    "Respond with ONLY one valid JSON object. No markdown fences, no prose, "
    "no explanations before or after the JSON. "
    "All Chinese text MUST be Traditional Chinese (繁體中文，臺灣用語)；"
    "absolutely no Simplified Chinese characters."
)

# 簡體字防線：LLM 偶爾混出簡體字（實測 qwen3 出過「调」）。
# 一對一無歧義的字元直接轉換；其餘僅偵測、命中就重試。
_SIMP_FIXTABLE = str.maketrans(
    "们调风设质实现经营销购价优级红蓝绿颜简适选择单图货轻软细节时气温润这让为会动点过还",
    "們調風設質實現經營銷購價優級紅藍綠顏簡適選擇單圖貨輕軟細節時氣溫潤這讓為會動點過還",
)
# 有歧義（发→發/髮、复→復/複…）不能盲轉的常見簡體字：只偵測
_SIMP_DETECT = set("发复头买卖东车长门问间业从体义乐应导热闹边")


def _fix_simplified(text: str) -> str:
    """安全子集直接轉正體。"""
    return text.translate(_SIMP_FIXTABLE)


def _has_simplified(text: str) -> bool:
    """是否仍殘留（無法安全自動轉換的）簡體字。"""
    return any(ch in _SIMP_DETECT for ch in text)


# ---------------------------------------------------------------------------
# 客戶端
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI 相容客戶端，封裝端點降級與結構化輸出驗證。"""

    def __init__(self) -> None:
        self._pickle = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or "EMPTY",  # SDK 不收空字串
            timeout=90,
            max_retries=0,
        )
        self._ollama = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=300,  # 14B 模型冷啟動 + 長輸出
            max_retries=0,
        )
        # 端點拒絕帶供應商前綴的 id（實測），兩個候選名都留著探測
        self._pickle_candidates = list(dict.fromkeys(
            [config.LLM_MODEL, config.LLM_MODEL.split("/")[-1]]
        ))
        self._pickle_model: str | None = None
        self._pickle_probed = False
        self._vision_ready: bool | None = None

    # -- 端點探測 -----------------------------------------------------------

    def _ensure_pickle(self) -> bool:
        """實測 Big Pickle 能否不帶 key 完成 chat.completions（只探一次）。"""
        if self._pickle_probed:
            return self._pickle_model is not None
        self._pickle_probed = True
        for name in self._pickle_candidates:
            try:
                self._pickle.chat.completions.create(
                    model=name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=8,
                )
                self._pickle_model = name
                return True
            except Exception:
                continue
        return False

    def _ensure_vision(self) -> bool:
        """確認 Ollama 視覺模型存在；不在就背景 pull，最多等 5 分鐘。"""
        if self._vision_ready is not None:
            return self._vision_ready
        model = config.OLLAMA_VISION_MODEL
        try:
            tags = {m.id for m in self._ollama.models.list().data}
        except Exception:
            self._vision_ready = False
            return False
        if model not in tags and model.split(":")[0] not in tags:
            try:
                proc = subprocess.Popen(
                    ["ollama", "pull", model],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                # 不殺 pull（可續傳），本次先放棄走手動輸入
                self._vision_ready = False
                return False
            except Exception:
                self._vision_ready = False
                return False
        self._vision_ready = True
        return True

    # -- 底層呼叫 -----------------------------------------------------------

    def _chat_once(self, client: OpenAI, model: str, messages: list[dict],
                   temperature: float) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content or ""
        content = _strip_think(content)
        if not content:
            # 推理模型把 token 預算燒在 reasoning 上時 content 會是空字串
            raise ValueError("空回覆")
        return content

    def _structured_call(self, system: str, user: str, schema: Type[T],
                         temperature: float = 0.6,
                         list_key: str | None = None,
                         n_items: int | None = None) -> list[T] | T | None:
        """端點鏈 + 重試 + pydantic 驗證。list_key 非 None 時回傳模型清單。

        每個端點最多 3 次嘗試（1 + 重試 2）；全部失敗回 None，由呼叫端套模板。
        """
        backends: list[tuple[OpenAI, str]] = []
        if self._ensure_pickle():
            backends.append((self._pickle, self._pickle_model))
        backends.append((self._ollama, config.OLLAMA_TEXT_MODEL))

        messages = [
            {"role": "system", "content": f"{system}\n{JSON_RULES} /no_think"},
            {"role": "user", "content": user},
        ]
        for client, model in backends:
            for attempt in range(3):
                try:
                    raw = self._chat_once(client, model, messages, temperature)
                    raw = _fix_simplified(raw)
                    if _has_simplified(raw):
                        raise ValueError("輸出含無法自動轉換的簡體字")
                    data = _extract_json(raw)
                    if list_key is not None:
                        items = data.get(list_key, data) if isinstance(data, dict) else data
                        if isinstance(items, dict):  # 模型只回單一物件時包成清單
                            items = [items]
                        parsed = [schema.model_validate(it) for it in items]
                        if not parsed:
                            raise ValueError("清單為空")
                        if n_items:
                            parsed = parsed[:n_items]
                        return parsed
                    return schema.model_validate(data)
                except Exception:
                    if attempt == 0 and client is client:  # 連線層錯誤也重試，保持簡單
                        pass
                    continue
        return None

    # -- 對外 API -----------------------------------------------------------

    def identify_product(self, image: Image.Image) -> ProductCard | None:
        """VLM 商品識別。模型/端點不可用回 None（UI 改走手動輸入）。"""
        if not self._ensure_vision():
            return None
        prompt = (
            "你是電商選品專家兼商品攝影總監。請辨識圖中商品，輸出 JSON："
            "category/name_guess/material/color/condition/target_audience 用繁體中文，"
            "product_class 用英文小寫。\n"
            '{"category": "商品類別", "name_guess": "推測商品名",'
            ' "material": "材質", "color": "主色",'
            ' "condition": "新舊狀況（如：二手，狀況良好，輕微使用痕跡）",'
            ' "selling_points": ["3-5 個賣點"], "target_audience": "目標客群",'
            ' "product_class": "rigid｜wearable｜handheld",'
            ' "worn_framing": "英文取景片語或空字串"}\n'
            "★product_class 是攝影策略關鍵，請依「這產品最自然的展示方式」判斷：\n"
            "  - wearable＝穿戴在身上才對（手錶/戒指/手環/項鍊/耳環/眼鏡/帽）。\n"
            "  - handheld＝拿在手中使用才對（遊戲手把/手機/筆/相機/滑鼠）。\n"
            "  - rigid＝放在檯面展示最對（香水/罐/瓶/皮夾/鞋/盒裝/3C 本體）。\n"
            "  判斷依據是「最自然的使用情境」，不是材質硬度。\n"
            "★worn_framing：若 wearable/handheld，請你決定「這產品該被戴在哪/怎麼握」的英文"
            "取景片語，要具體寫出身體部位，例：'a person's wrist wearing the watch, forearm "
            "visible' / 'two hands holding the game controller while playing' / "
            "'a hand wearing the ring, fingers visible'；若 rigid 則留空字串。\n" + JSON_RULES
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": _image_to_data_url(image)}},
            ],
        }]
        for _ in range(3):
            try:
                raw = self._chat_once(
                    self._ollama, config.OLLAMA_VISION_MODEL, messages, 0.3)
                return ProductCard.model_validate(_extract_json(raw))
            except Exception:
                continue
        return None

    def locate_worn_region(self, scene_image: Image.Image, product_desc: str,
                           body_part: str = "手腕") -> list | None:
        """VLM 看裸身體部位場景 → 回「產品該放的矩形框」(x0,y0,x1,y1，0-1000 正規化)。
        比膚色 CV 可靠：VLM 懂手錶該戴腕部、比例約腕寬。失敗回 None（pipeline 退回膚色偵測）。"""
        if not self._ensure_vision():
            return None
        prompt = (
            f"這張圖是裸露的{body_part}（還沒戴任何東西）。我要把一個「{product_desc}」"
            f"自然地放上去（像真的戴著/握著）。\n"
            '只輸出 JSON：{"found": true, "box": [x0, y0, x1, y1]}\n'
            "box＝該產品**應佔據的矩形**，座標 0-1000 正規化（左上 0,0、右下 1000,1000）。\n"
            "★務必符合真實比例：例如手錶錶體寬度約等於手腕寬度，**不要佔滿整條手臂**；"
            "框要正好對準該戴/握的部位。找不到合適部位回 {\"found\": false}。" + JSON_RULES
        )
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(scene_image)}}]}]
        for _ in range(3):
            try:
                raw = self._chat_once(self._ollama, config.OLLAMA_VISION_MODEL, messages, 0.2)
                d = _extract_json(raw)
                box = d.get("box")
                if d.get("found") and isinstance(box, list) and len(box) == 4:
                    return [float(v) for v in box]
                return None
            except Exception:
                continue
        return None

    def judge_worn(self, result_image: Image.Image, product_desc: str) -> dict | None:
        """VLM 評穿戴/手持成品：自然度/大小/位置/破綻 → 驅動重跑或收。失敗回 None。"""
        if not self._ensure_vision():
            return None
        prompt = (
            f"這是一張「{product_desc}」戴/握在身上的廣告圖。請當嚴格的廣告美術總監評估。\n"
            '只輸出 JSON：{"score": 1-10, "size": "too_big｜ok｜too_small", '
            '"placement": "ok｜off", "natural": true/false, "issues": ["..."]}\n'
            "size＝產品相對身體部位的比例；placement＝有沒有戴/握在對的位置；"
            "issues＝最該修的問題（繁體中文）。" + JSON_RULES
        )
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(result_image)}}]}]
        for _ in range(3):
            try:
                raw = self._chat_once(self._ollama, config.OLLAMA_VISION_MODEL, messages, 0.3)
                d = _extract_json(raw)
                return d if isinstance(d, dict) else None
            except Exception:
                continue
        return None

    def plan_scenes(self, card: ProductCard, user_brief: str,
                    n: int = 3, lifestyle: bool = False,
                    mode: str = "locked") -> list[ScenePlan]:
        """場景企劃：商品卡 + 口語需求 → N 組可直接餵 diffusion 的方案。

        mode="reshape"：穿戴/手持重塑模式，scene_prompt 只描述背景情境＋光線
            （戴/握的取景由 reshape.py 依品類補），不需擺位。
        lifestyle=False：乾淨電商商品背景（無人物）。
        lifestyle=True：情境廣告照，忠實納入使用者描述的人物/模特/場景元素。
        """
        if mode == "reshape":
            system = (
                "You are an advertising art director for WORN/HANDHELD product ads. "
                "The product (a watch, ring, controller, etc.) will be re-rendered "
                "being worn or held by a person via an image-conditioned model. "
                "You only design the BACKGROUND CONTEXT and lighting around that pose."
            )
            scene_rule = (
                "- scene_prompt：英文，只寫**背景情境＋光氛**（不要描述產品本身、不要寫人物"
                "長相、不要寫擺位）。例：'in a cozy minimalist cafe, warm morning window "
                "light, blurred background' / 'on a city street at golden hour, bokeh "
                "lights' / 'against a clean studio grey backdrop, soft beauty light' / "
                "'luxury marble interior, dramatic side light'。\n"
                "- ★N 組的場景情境與光氛要**明顯不同**（咖啡廳/街頭/工作室/戶外/居家/精品店…），"
                "色調氛圍各異，務必精簡（**25 個英文字以內**）。\n"
                "- product_scale/product_x/product_y 填預設即可（重塑模式不使用擺位）。\n"
                "- negative_prompt：英文（lowres, deformed hands, blurry, distorted）\n"
            )
        elif lifestyle:
            system = (
                "You are an advertising art director. The product is locked in "
                "place and the scene is generated AROUND it by inpainting. Create a "
                "vivid LIFESTYLE advertising scene that faithfully includes whatever "
                "the user describes — including people, models, settings, activities."
            )
            scene_rule = (
                "- 你是廣告美術指導，要**自己決定構圖**：產品放前景的「左側或右側」"
                "（product_x 選 0.24~0.30 或 0.70~0.76，二擇一），但**比例要真實、不可"
                "巨大**（product_scale 0.30~0.38，相當於放在桌上或手邊的真實飲料罐尺寸，"
                "切忌跟人一樣高），product_y 0.66~0.74。\n"
                "- scene_prompt：英文 lifestyle 廣告情境，**務必明確把人物/模特放在產品的"
                "『對側』背景**（產品在右就寫 a model on the LEFT side in the background；"
                "產品在左就寫 on the RIGHT side），人物**較小、在中後景，絕不可擋住或壓在"
                "產品上**。忠實納入使用者描述的元素（比基尼、海、活動等），結尾加 "
                "lifestyle advertising photography, cinematic, vivid colors。"
                "不要描述商品本身（已鎖定）。\n"
                "- negative_prompt：英文（lowres, deformed, bad hands, "
                "product covering person, multiple products；**不要**排除 people）\n"
            )
        else:
            system = (
                "You are an art director for e-commerce product photography. "
                "The product is locked in place and the scene is generated AROUND it "
                "by inpainting. Your scene_prompt must describe a clean PRODUCT "
                "BACKDROP — a surface the product sits ON plus a simple background — "
                "never a full scene with its own subject."
            )
            scene_rule = (
                "- 你是商品攝影美術指導，**自己決定構圖**：產品當醒目英雄、置中偏下"
                "（product_x 約 0.5、product_y 0.66~0.72、product_scale 0.50~0.60，"
                "主體要大不要小），讓上方有背景、下方有**實體檯面**讓產品踩穩。\n"
                "- ★ 接地（整夜第1-2批最大教訓，純棚拍最會浮）：scene_prompt 必須寫"
                "產品站在**具體實體表面**（marble/oak wood/concrete/brushed metal），"
                "並寫 'soft diffused contact shadow, short, ambient occlusion at the base, "
                "shadow darkest where it meets the surface'。★陰影方向必須**與主光同一側、"
                "單一光源邏輯**（light_direction 同時驅動 key 光與投影），禁止長硬投影或"
                "雙重/反向陰影。\n"
                "- ★ brief 鎖定：純白要寫 'pure white RGB 255 255 255 seamless, zero "
                "gradient'；指定光線/材質要寫死具體（warm amber sunset / hard spotlight "
                "on black / coarse linen weave），別只用形容詞（模型會回灰漸層預設）。\n"
                "- scene_prompt：英文 SDXL prompt。必須是「商品擺台＋豐富但乾淨的背景」，"
                "務必寫出產品所站的**檯面/平面**（marble countertop, wooden table, "
                "stone podium…）與後方**有層次的虛化背景**（如 blurred sunny beach with "
                "sea and palm bokeh / cozy cafe interior），避免死板純色或一片空沙；"
                "結尾加 professional product photography, soft shadows, shallow depth "
                "of field, vibrant。\n"
                "  ★ 嚴禁：人物、其他同類商品、街景車輛、搶鏡主體。背景要美但不雜亂。\n"
                "  ★ 不要描述商品本身（已鎖定）。\n"
                "- negative_prompt：英文（cluttered, busy, extra objects, people, "
                "multiple products, duplicate cans, flat plain background, dull）\n"
                "- ★【角度多樣】product_view 三選一（front／three_quarter／top）。"
                "N 組**不要都 front**：至少 1 組 three_quarter（3/4 側角，最有立體質感）、"
                "可安排 1 組 top（俯視平拍 flat-lay，配桌面散落小道具）。讓主角角度有變化。\n"
                "- ★【背景多樣】N 組的環境類型要**明顯不同**，禁止都同一種檯面＋灰漸層："
                "例如咖啡廳木桌／工業風水泥／大理石浴室檯／木質書桌暖光／戶外石階草地／"
                "霓虹夜店等各挑不同的，色調與光氛也要區隔。\n"
            )
        user = (
            f"商品卡：{card.model_dump_json()}\n"
            f"使用者需求：{user_brief or '（無，請自由發揮）'}\n\n"
            f"請給出 {n} 組風格明顯不同的場景方案，輸出 JSON："
            '{"plans": [{"plan_name": "...", "scene_prompt": "...",'
            ' "negative_prompt": "...", "light_direction": "...",'
            ' "light_desc": "...", "mood": "...", "composition_tip": "...",'
            ' "product_scale": 0.42, "product_x": 0.5, "product_y": 0.66,'
            ' "product_view": "three_quarter"}]}\n'
            "規則：\n"
            + scene_rule +
            "- ★ scene_prompt 務必精簡（**45 個英文字以內**，CLIP 只讀前 77 token，"
            "過長後段會被丟棄），把最重要的元素寫在最前面，並務必包含 "
            "'a clear soft contact shadow grounding the product'（接地陰影避免浮貼）。\n"
            "- light_desc：英文光線描述（方向、色溫、軟硬，精簡）\n"
            "- light_direction：只能是 left/right/top/bottom/front/back 之一，"
            "且必須與 light_desc 一致\n"
            "- plan_name/mood/composition_tip：繁體中文"
        )
        plans = self._structured_call(system, user, ScenePlan, temperature=0.8,
                                      list_key="plans", n_items=n)
        if plans:
            # 不足 n 組時以預設模板補滿
            i = 0
            while len(plans) < n:
                plans.append(DEFAULT_SCENE_PLANS[i % len(DEFAULT_SCENE_PLANS)])
                i += 1
            return plans
        return [DEFAULT_SCENE_PLANS[i % len(DEFAULT_SCENE_PLANS)]
                for i in range(n)]

    def write_copy(self, card: ProductCard, plan: ScenePlan) -> CopyPack:
        """文案生成：商品卡 + 選定方案 → 蝦皮標題 / 賣點 / IG 貼文。"""
        system = "你是台灣電商文案寫手，文字自然有溫度，不浮誇。全部使用繁體中文。"
        user = (
            f"商品卡：{card.model_dump_json()}\n"
            f"視覺方案氛圍：{plan.plan_name}／{plan.mood}\n\n"
            "請輸出 JSON："
            '{"shopee_title": "蝦皮標題，25-40 字，含搜尋關鍵字與【】標記",'
            ' "bullet_points": ["五點賣點，每點 15 字內"],'
            ' "ig_caption": "IG 貼文，2-4 句，含 emoji，呼應視覺氛圍",'
            ' "hashtags": ["5-8 個繁體中文 hashtag，含 # 字號"]}'
        )
        result = self._structured_call(system, user, CopyPack, temperature=0.7)
        return result if result is not None else _default_copy(card)

    def parse_edit(self, plan: ScenePlan, instruction: str) -> ScenePlan:
        """多輪修改：「光暖一點、換大理石」→ 修改後的完整方案。

        簽名依 pipeline.py 整合契約：parse_edit(plan, instruction)。
        解析失敗回傳原方案（不動為安全預設）。
        """
        system = (
            "You revise a product-photography scene plan according to the "
            "user's spoken instruction. Change ONLY what the instruction "
            "implies; keep all other fields identical."
        )
        user = (
            f"目前方案 JSON：{plan.model_dump_json()}\n"
            f"使用者修改指令：{instruction}\n\n"
            "請輸出修改後的完整方案 JSON（欄位同上）。規則：\n"
            "- scene_prompt/negative_prompt/light_desc 保持英文\n"
            "- light_direction 只能是 left/right/top/bottom/front/back\n"
            "- plan_name/mood/composition_tip 保持繁體中文\n"
            "- 指令沒提到的欄位原封不動"
        )
        result = self._structured_call(system, user, ScenePlan, temperature=0.3)
        return result if result is not None else plan

    def release_models(self) -> None:
        """請 Ollama 立即卸載已載入的模型，把 VRAM 讓給 diffusion。

        qwen3:14b（約 11GB）若與 SDXL+IC-Light（約 15GB）同駐會爆 24GB，
        pipeline 在 LLM 階段結束後呼叫本方法；失敗不拋例外（無傷大雅）。
        """
        base = config.OLLAMA_BASE_URL.rsplit("/v1", 1)[0]
        try:
            with urllib.request.urlopen(f"{base}/api/ps", timeout=5) as resp:
                loaded = [m["name"] for m in json.load(resp).get("models", [])]
        except Exception:
            return
        for name in loaded:
            try:
                req = urllib.request.Request(
                    f"{base}/api/generate",
                    data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=30).read()
            except Exception:
                pass
