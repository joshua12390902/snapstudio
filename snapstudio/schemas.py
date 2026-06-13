"""LLM 結構化輸出契約（ARCHITECTURE.md 第 4 節）。

三個模型對應三個 JSON 契約：商品卡 / 場景方案 / 文案包。
欄位語言約定：scene_prompt、negative_prompt、light_desc 為英文（直接餵
diffusion），其餘描述性欄位為繁體中文。
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LightDirection = Literal["left", "right", "top", "bottom", "front", "back"]
# 多角度輸入：每個場景方案指定要套用哪張角度照（使用者可上傳多角度真實照片）
ProductView = Literal["front", "three_quarter", "top"]
# 產品類別 → 決定渲染模式：rigid=鎖定模式(像素精準)、wearable/handheld=重塑模式(可戴/握)
ProductClass = Literal["rigid", "wearable", "handheld"]


class ProductCard(BaseModel):
    """商品卡：VLM 商品識別輸出（4.1）。"""

    category: str = Field(description="商品類別，繁體中文")
    name_guess: str = Field(description="推測商品名稱，繁體中文")
    material: str = Field(description="材質，繁體中文")
    color: str = Field(description="主色，繁體中文")
    condition: str = Field(description="新舊狀況描述，繁體中文")
    selling_points: list[str] = Field(default_factory=list, description="賣點清單")
    target_audience: str = Field(default="", description="目標客群，繁體中文")
    # 渲染模式自動路由依據：rigid→鎖定模式、wearable/handheld→重塑模式
    product_class: ProductClass = Field(
        default="rigid", description="rigid 剛性擺台 / wearable 穿戴 / handheld 手持")
    # 重塑取景：VLM 決定「這產品怎麼戴/握」的英文片語（rigid 留空），交給 LLM 判斷而非硬寫
    worn_framing: str = Field(
        default="", description="英文，穿戴/手持取景片語，如 a person's wrist wearing the watch")
    # 最佳呈現：VLM 決定走「乾淨商品擺台(clean)」還是「穿戴/手持(worn)」。穿戴對腳/耳/複雜
    # 姿態生成不可靠，VLM 應只在穿戴是該品類強烈慣例且身體部位簡單可靠時才選 worn，否則 clean。
    best_shot: str = Field(
        default="clean", description="clean 乾淨擺台 / worn 穿戴或手持，VLM 依可靠度與慣例決定")

    @field_validator("best_shot", mode="before")
    @classmethod
    def _norm_shot(cls, v: object) -> object:
        return "worn" if isinstance(v, str) and "worn" in v.strip().lower() else "clean"

    @field_validator("product_class", mode="before")
    @classmethod
    def _norm_class(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        s = v.strip().lower()
        if any(k in s for k in ("wear", "worn", "jewel", "accessor")):
            return "wearable"
        if any(k in s for k in ("handheld", "hand-held", "hand_held", "handhold",
                                "held", "hand")):
            return "handheld"
        return "rigid"


class ScenePlan(BaseModel):
    """場景方案：LLM 場景企劃輸出（4.2），一次產 N 組。"""

    plan_name: str = Field(description="方案名稱，繁體中文")
    scene_prompt: str = Field(description="SDXL 場景 prompt，英文")
    negative_prompt: str = Field(default="", description="SDXL 負面詞，英文")
    light_direction: LightDirection = Field(description="光源方向（IC-Light 條件）")
    light_desc: str = Field(description="光線描述，英文，併入 relight prompt")
    mood: str = Field(default="", description="氛圍，繁體中文")
    composition_tip: str = Field(default="", description="構圖建議，繁體中文")
    # AI 自動決定的主角擺位（美術指導角色）：免使用者手動拉滑桿
    product_scale: float = Field(default=0.42, ge=0.2, le=0.6,
                                 description="產品最長邊佔畫面比例")
    product_x: float = Field(default=0.5, ge=0.1, le=0.9,
                             description="產品中心水平位置 0-1")
    product_y: float = Field(default=0.66, ge=0.4, le=0.85,
                             description="產品中心垂直位置 0-1")
    # 多角度：本方案想用的產品視角；pipeline 依此挑使用者上傳的對應角度照，
    # 沒提供該角度時自動退回正面。讓 N 組成品的主角角度有變化。
    product_view: ProductView = Field(
        default="front", description="產品視角 front/three_quarter/top")

    @field_validator("product_view", mode="before")
    @classmethod
    def _norm_view(cls, v: object) -> object:
        """把 LLM 各種講法正規化成三類視角；無法判讀時退回 front。"""
        if not isinstance(v, str):
            return v
        s = v.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
        if any(k in s for k in ("top", "overhead", "above", "flat_lay", "flatlay", "birds")):
            return "top"
        if any(k in s for k in ("three_quarter", "3_4", "34", "quarter", "angle",
                                "side", "profile", "left", "right", "diagonal")):
            return "three_quarter"
        return "front"

    @field_validator("product_scale", "product_x", "product_y", mode="before")
    @classmethod
    def _coerce_float(cls, v: object) -> object:
        """LLM 偶爾回字串或百分比（如 "50%"）；去掉 % 後轉 float、>1 視為百分比。"""
        try:
            s = str(v).strip().rstrip("%").strip() if isinstance(v, str) else v
            f = float(s)
            return f / 100.0 if f > 1.0 else f
        except (TypeError, ValueError):
            return v

    @field_validator("light_direction", mode="before")
    @classmethod
    def _norm_direction(cls, v: object) -> object:
        """LLM 偶爾回大寫或帶空白的方向值，先正規化再交給 Literal 驗證。"""
        if isinstance(v, str):
            return v.strip().lower()
        return v


class CopyPack(BaseModel):
    """文案包：LLM 文案輸出（4.3）。"""

    shopee_title: str = Field(description="蝦皮標題，含關鍵字，繁體中文")
    bullet_points: list[str] = Field(default_factory=list, description="五點賣點")
    ig_caption: str = Field(default="", description="IG 貼文，可含 emoji")
    hashtags: list[str] = Field(default_factory=list, description="hashtag 清單")
