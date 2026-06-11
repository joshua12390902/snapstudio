"""SnapStudio 編排器（v2，inpaint-grounded）：
去背 → 商品識別 → 場景企劃 → **鎖產品 inpaint 生成周圍場景** → 鎖回產品像素
→（選配）IC-Light 光線融合 → 文案。

v2 重設計（解決舊版「貼紙感」）：
- 舊版先生成完整背景再把產品貼上去 latent 串接重打光 → 產品浮貼、不接地。
- 新版把產品鎖在畫布定位，用 SDXL inpaint 在它「周圍」生成檯面與場景，
  接地陰影、反光在同一次去噪自然長出（groundgen）；產品像素最後原樣貼回
  （電商鐵則：商品本體不可被模型篡改）。
- 使用者可調主角的位置/角度/大小（compose.Placement）。
- IC-Light（本專題自製 diffusers 0.39 pipeline）降為選配的全域光線融合層，
  讓產品表面光照與場景一致；預設精緻檔開、快速檔關。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from PIL import Image

from . import config  # noqa: F401  # 匯入即設定 HF_HUB_OFFLINE 等環境變數
from .compose import Placement, build_scene_inputs, paste_back
from .llm import CopyPack, LLMClient, ProductCard, ScenePlan, fallback_product_class

ProgressCB = Optional[Callable[[str, float], None]]

# 品質檔位 → (inpaint 步數, guidance)。預設只用 fine；fast 保留供 CLI/相容。
# 數值經三輪審查員遞迴優化定案：步數 32 出細節與接地、guidance 7.5 兼顧質感與
# 避免高 guidance 的過飽和/藍綠 color bleeding。
_QUALITY = {"fine": (32, 7.5), "fast": (14, 7.5)}
CANVAS = (1024, 1024)


@dataclass
class StudioResult:
    """一次完整生成的素材包。"""
    card: ProductCard
    plans: list[ScenePlan]
    shots: list[Image.Image]
    copy: Optional[CopyPack]
    timings: dict
    fg: Optional[Image.Image] = field(default=None)        # 去背 RGBA（正面，refine 重跑用）
    previews: list = field(default_factory=list)           # 主角擺放預覽（展示中間產物）
    placement: Placement = field(default_factory=Placement)
    views: dict = field(default_factory=dict)              # {"front": 去背RGBA}（相容用）
    view_pool: list = field(default_factory=list)          # 去背RGBA角度池（多張照輪流用）
    mode: str = "locked"                                   # locked 鎖定 / reshape 重塑
    product_class: str = "rigid"                           # rigid / wearable / handheld


class SnapStudio:
    """單卡（RTX 3090）編排器。quality: "fast"=12 步 / "fine"=20 步。"""

    BASE_SEED = 42

    def __init__(self, quality: str = "fine"):
        if quality not in _QUALITY:
            raise ValueError(f"quality 必須是 'fast' 或 'fine'，收到 {quality!r}")
        self.quality = quality
        self.llm = LLMClient()
        self._matting = None
        self._inpaint = None
        self._relight = None
        self._reshape = None

    def set_quality(self, quality: str) -> None:
        if quality not in _QUALITY:
            raise ValueError(f"quality 必須是 'fast' 或 'fine'，收到 {quality!r}")
        self.quality = quality

    # ---------- 模型載入 ----------

    def _ensure_matting(self):
        if self._matting is None:
            from .matting import Matting
            self._matting = Matting()

    def _ensure_inpaint(self):
        if self._inpaint is None:
            # 切回鎖定模式時卸掉重塑 pipeline 釋放 VRAM
            if self._reshape is not None:
                self._reshape.unload()
                self._reshape = None
            from .groundgen import SceneInpainter
            self._inpaint = SceneInpainter(accelerate=False)  # 標準步數品質優於 LCM

    def _ensure_relight(self):
        if self._relight is None:
            from .relight import Relighter
            self._relight = Relighter(mode="fbc", quality="fast")

    def _ensure_reshape(self):
        if self._reshape is None:
            # 重塑與鎖定共用單卡，先卸掉 inpaint 與 relight 釋放 VRAM（重塑都用不到）
            if self._inpaint is not None:
                self._inpaint.unload()
                self._inpaint = None
            if self._relight is not None:
                self._relight.unload()
                self._relight = None
            from .reshape import ReshapeStudio
            self._reshape = ReshapeStudio()

    @staticmethod
    def _notify(cb: ProgressCB, stage: str, frac: float) -> None:
        if cb is None:
            return
        try:
            cb(stage, min(max(frac, 0.0), 1.0))
        except Exception:
            pass

    @staticmethod
    def _fallback_card(user_brief: str) -> ProductCard:
        name = (user_brief or "").strip() or "商品"
        return ProductCard(
            category="一般商品", name_guess=name[:50], material="未知", color="未知",
            condition="狀況良好", selling_points=["實物拍攝", "品質可靠", "快速出貨"],
            target_audience="一般消費者",
        )

    # ---------- 單方案渲染 ----------

    @staticmethod
    def _pick_view(views: dict, requested: str) -> Image.Image:
        """依方案要的視角挑去背圖；沒提供該角度 → three_quarter → front 退回。"""
        for key in (requested, "three_quarter", "front"):
            if views.get(key) is not None:
                return views[key]
        # 理論上 front 一定有；保底回任一張
        return next(iter(views.values()))

    @staticmethod
    def _plan_placement(plan: ScenePlan, override: Placement | None) -> Placement:
        """擺位優先序：使用者手動覆蓋 > LLM 美術指導決定 > 預設。"""
        if override is not None:
            return override.clamped()
        return Placement(scale=plan.product_scale, cx=plan.product_x,
                         cy=plan.product_y).clamped()

    def _render_plan(self, fg: Image.Image, plan: ScenePlan, placement: Placement | None,
                     seed: int, harmonize: bool, allow_people: bool = False
                     ) -> tuple[Image.Image, Image.Image, dict]:
        """鎖產品 inpaint 生成場景 → 貼回產品 →（選配）IC-Light 光線融合。

        placement=None 時用 LLM 在 plan 裡決定的擺位（AI 自動構圖）。
        回傳 (成品, 擺放預覽, 計時)。
        """
        steps, guidance = _QUALITY[self.quality]
        t = {}
        place = self._plan_placement(plan, placement)
        parts = build_scene_inputs(fg, CANVAS, place, seed=seed)

        t0 = time.time()
        gen = self._inpaint.generate(
            parts["init"], parts["mask"], plan.scene_prompt,
            negative_prompt=plan.negative_prompt, steps=steps,
            guidance_scale=guidance, seed=seed, allow_people=allow_people,
        )
        shot = paste_back(gen, parts["product"], light_direction=plan.light_direction)
        t["inpaint"] = round(time.time() - t0, 2)

        if harmonize:
            # IC-Light fbc：前景＝**依相同擺放對齊**的灰底產品圖（product_gray，RGB），
            # 背景＝inpaint 成品；前景位置與場景中的產品一致，避免「重疊」。
            t0 = time.time()
            self._ensure_relight()
            relit = self._relight.relight(
                parts["product_gray"].resize((768, 768)),
                shot.resize((768, 768)),
                prompt=f"{plan.scene_prompt}, {plan.light_desc}",
                width=768, height=768, hires=False, lcm=True, seed=seed,
            ).resize(CANVAS)
            # 融光只取「場景/環境光」，最後把銳利的原始產品像素再貼回一次，
            # 確保主角邊界永遠像素級俐落（IC-Light 768 會糊化產品）。
            shot = paste_back(relit, parts["product"], light_direction=plan.light_direction)
            t["harmonize"] = round(time.time() - t0, 2)

        return shot, parts["preview"], t

    def _render_reshape(self, ref_rgba: Image.Image, plan: ScenePlan, framing: str,
                        seed: int) -> tuple[Image.Image, dict]:
        """重塑模式單方案：IP-Adapter 把產品重畫成 framing 的穿戴/手持姿態 + plan 場景。"""
        t = {}
        t0 = time.time()
        shot = self._reshape.generate(
            ref_rgba, framing, plan.scene_prompt,
            negative_prompt=plan.negative_prompt, seed=seed,
        )
        t["reshape"] = round(time.time() - t0, 2)
        return shot, t

    # ---------- 對外 ----------

    def process(self, image: Image.Image, user_brief: str = "", n_plans: int = 3,
                progress_cb: ProgressCB = None, manual_desc: str | None = None,
                placement: Placement | None = None, harmonize: bool | None = None,
                lifestyle: bool = False, angle_images: dict | None = None,
                mode: str = "auto",
                ) -> StudioResult:
        """一鍵全流程：照片 → N 組整合場景成品 + 文案包。

        mode：渲染模式路由。"auto"＝由 VLM 判斷的 product_class 自動決定
            （rigid→鎖定模式像素精準、wearable/handheld→重塑模式戴/握上身）；
            也可手動 "locked"／"reshape" 覆蓋。
        placement：主角擺放手動覆蓋；**None＝由 LLM 美術指導自動決定構圖**（預設）。
        harmonize：是否做 IC-Light 光線融合；None 時精緻檔開、快速檔關。
        lifestyle：情境廣告照（允許人物/模特、忠實照使用者描述）。
        angle_images：額外角度照「角度池」。可為 list[PIL]（多圖上傳，推薦）或
            dict（相容舊版）。連同主圖各去背一次組成 view_pool，N 組方案輪流取用
            （pool[i % len]），丟幾張就有幾種角度，不必標哪張是什麼角度。
        """
        place_override = placement.clamped() if placement is not None else None
        if harmonize is None:
            harmonize = self.quality == "fine" and not lifestyle  # 情境照預設不融光（保銳利人景）
        t_total = time.time()
        timings: dict = {}

        # 先卸掉上一輪可能殘留在顯卡的 Ollama 模型（VLM 裁判跑完未必已釋放），
        # 否則接著的 SAM2 去背會在滿載的 24GB 上 OOM、退回較糙的 rembg 邊緣。
        self.llm.release_models(wait=True)
        self._notify(progress_cb, "去背中", 0.02)
        t0 = time.time()
        self._ensure_matting()
        fg = self._matting.cutout(image.convert("RGB"))
        # 多角度「角度池」：主圖 + 每張額外角度照各去背一次，組成池子（位置無關）。
        # angle_images 可為 list（多圖上傳）或 dict（相容舊呼叫）。
        view_pool = [fg]
        if angle_images:
            extras = (list(angle_images.values()) if isinstance(angle_images, dict)
                      else list(angle_images))
            for img in extras:
                if img is None:
                    continue
                try:
                    view_pool.append(self._matting.cutout(img.convert("RGB")))
                except Exception:  # noqa: BLE001  # 單張角度失敗不該拖垮整體
                    pass
        views = {"front": fg}
        timings["matting"] = round(time.time() - t0, 2)

        self._notify(progress_cb, "商品識別中", 0.08)
        t0 = time.time()
        vlm_card = None
        if manual_desc:
            card = self._fallback_card(manual_desc)
        else:
            vlm_card = self.llm.identify_product(image)
            card = vlm_card or self._fallback_card(user_brief)
        # AI 自判產品類別與模式：VLM 在線→用 VLM 的 product_class；否則關鍵字備援。
        # 保底：class 為 rigid 時（含 VLM 回了表外字串被壓成 rigid）再跑一次關鍵字補判，
        # 避免手錶/戒指等穿戴品因 VLM 用詞不在表內而誤路由到鎖定模式。
        if vlm_card is not None:
            product_class = vlm_card.product_class
            if product_class == "rigid":
                product_class = fallback_product_class(vlm_card)
        else:
            product_class = fallback_product_class(card)
        if mode == "auto":
            resolved_mode = "reshape" if product_class in ("wearable", "handheld") else "locked"
        else:
            resolved_mode = mode
            if resolved_mode == "reshape" and product_class == "rigid":
                product_class = "handheld"  # 使用者強制重塑但類別為剛性 → 給手持取景
        timings["identify"] = round(time.time() - t0, 2)

        self._notify(progress_cb, "場景企劃中", 0.14)
        t0 = time.time()
        plans = list(self.llm.plan_scenes(
            card, user_brief=user_brief, n=n_plans,
            lifestyle=(lifestyle and resolved_mode == "locked"),
            mode="reshape" if resolved_mode == "reshape" else "locked",
        ))[:n_plans]
        timings["plan"] = round(time.time() - t0, 2)

        self._notify(progress_cb, "文案生成中", 0.20)
        t0 = time.time()
        copy_pack = self.llm.write_copy(card, plans[0]) if plans else None
        timings["copy"] = round(time.time() - t0, 2)
        self.llm.release_models()  # 讓出 Ollama 佔的 VRAM 再載 diffusion

        shots, previews = [], []
        n = max(len(plans), 1)

        if resolved_mode == "reshape":
            from .reshape import (framing_for, composite_real_face, bare_scene_prompt,
                                  compact_reference, BARE_SCENE_NEGATIVE)
            from . import wornplace
            framing = framing_for(card, product_class)  # 取景由 VLM 決定（worn_framing）
            use_anydoor = wornplace.available()
            self._notify(progress_cb,
                         "載入場景模型（AnyDoor 路線）" if use_anydoor
                         else "載入重塑模型（IP-Adapter）", 0.26)
            t0 = time.time()
            self._ensure_reshape()
            timings["load_models"] = round(time.time() - t0, 2)

            if use_anydoor:
                # 1) text2img 生 N 張「裸身體部位」場景
                scenes = []
                seeds = [self.BASE_SEED + i for i in range(len(plans))]
                for i, plan in enumerate(plans):
                    self._notify(progress_cb, f"生成裸場景 {i + 1}/{len(plans)}",
                                 0.30 + 0.30 * i / n)
                    sp = bare_scene_prompt(framing, plan.scene_prompt, card.category)
                    scenes.append(self._reshape.generate_scene(
                        sp, negative_prompt=BARE_SCENE_NEGATIVE, seed=seeds[i]))
                # 2) 卸 RealVisXL 釋 VRAM → AnyDoor subprocess 自動擺放（朝向/光照/環繞）
                self._reshape.unload(); self._reshape = None
                # AnyDoor 用「精簡參考」（裁到產品主面，去掉折疊錶帶等拖尾→不再生雙錶）；
                # 真實細節仍由 composite_real_face 用完整 fg 補回。
                ref_anydoor = compact_reference(fg)
                # 膚色偵測身體部位定擺放遮罩（實測比 VLM bounding-box grounding 準）
                masks = [wornplace.body_part_mask(s, product_class) for s in scenes]
                self._notify(progress_cb, "AnyDoor 自動擺放（朝向/光照/環繞）", 0.62)
                t0 = time.time()
                try:
                    placed = wornplace.place_batch(ref_anydoor, scenes, seeds, masks=masks)
                except Exception:  # noqa: BLE001  # AnyDoor 失敗不可崩，退回裸場景
                    placed = [None] * len(scenes)
                timings["anydoor"] = round(time.time() - t0, 2)
                # 3) 把真實平面細節面（錶盤/標籤）合成回去 → 真實 logo
                for i, p in enumerate(placed):
                    self._notify(progress_cb, f"真實細節合成 {i + 1}/{len(plans)}",
                                 0.68 + 0.20 * i / n)
                    shots.append(composite_real_face(p, fg) if p is not None else scenes[i])
                    previews.append(fg)
                # 4) VLM 裁判（定性=VLM強項）：判定「太大」的，自動以更小遮罩重跑一次
                pdesc = card.name_guess or card.category or "產品"
                self._notify(progress_cb, "VLM 品質裁判", 0.90)
                retry = []  # (idx, scene, seed, smaller_mask)
                for i, p in enumerate(placed):
                    if p is None:
                        continue
                    v = self.llm.judge_worn(shots[i], pdesc)
                    if v and v.get("size") == "too_big":
                        retry.append((i, scenes[i], seeds[i],
                                      wornplace.body_part_mask(scenes[i], product_class, scale=0.34)))
                if retry:
                    self.llm.release_models()  # 卸 VLM 再交給 AnyDoor subprocess
                    self._notify(progress_cb,
                                 f"VLM 判 {len(retry)} 張過大 → 縮小重跑", 0.94)
                    try:
                        re_placed = wornplace.place_batch(
                            ref_anydoor, [r[1] for r in retry], [r[2] for r in retry],
                            masks=[r[3] for r in retry])
                    except Exception:  # noqa: BLE001
                        re_placed = [None] * len(retry)
                    for (idx, *_), rp in zip(retry, re_placed):
                        if rp is not None:
                            shots[idx] = composite_real_face(rp, fg)
            else:
                for i, plan in enumerate(plans):
                    self._notify(progress_cb,
                                 f"重塑生成 {i + 1}/{len(plans)}：{plan.plan_name}",
                                 0.30 + 0.68 * i / n)
                    shot, t = self._render_reshape(fg, plan, framing, self.BASE_SEED + i)
                    shot = composite_real_face(shot, fg)  # 真實平面細節合成
                    for k, v in t.items():
                        timings[f"{k}_{i + 1}"] = v
                    shots.append(shot)
                    previews.append(fg)
        else:
            self._notify(progress_cb, "載入生成模型", 0.26)
            t0 = time.time()
            self._ensure_inpaint()
            timings["load_models"] = round(time.time() - t0, 2)
            for i, plan in enumerate(plans):
                base = 0.30 + 0.68 * i / n
                self._notify(progress_cb,
                             f"生成場景 {i + 1}/{len(plans)}：{plan.plan_name}", base)
                fg_view = view_pool[i % len(view_pool)]  # 角度池輪流→各方案不同角度
                shot, preview, t = self._render_plan(
                    fg_view, plan, place_override, self.BASE_SEED + i, harmonize,
                    allow_people=lifestyle,
                )
                for k, v in t.items():
                    timings[f"{k}_{i + 1}"] = v
                shots.append(shot)
                previews.append(preview)
            # VLM 品質把關（生圖也跟 VLM 溝通修正）：卸 inpaint→逐張判 AI 破綻→
            # 有瑕疵就「換 seed＋把瑕疵加進負面詞」重生（自動化人工目視）
            pdesc = card.name_guess or card.category or "產品"
            self._notify(progress_cb, "VLM 品質把關", 0.86)
            self._inpaint.unload(); self._inpaint = None
            fixes = []
            for i in range(len(plans)):
                v = self.llm.judge_product_shot(shots[i], pdesc)
                if v and v.get("needs_fix"):
                    p2 = plans[i].model_copy()
                    flaw = (v.get("flaw_terms") or "").strip()
                    if flaw:
                        p2.negative_prompt = (plans[i].negative_prompt + ", " + flaw).strip(", ")
                    fixes.append((i, p2))
            if fixes:
                self.llm.release_models()  # 卸 VLM 再重載 inpaint
                self._ensure_inpaint()
                self._notify(progress_cb,
                             f"VLM 判 {len(fixes)} 張有瑕疵 → 換 seed＋補負面詞重生", 0.94)
                for i, p2 in fixes:
                    fg_view = view_pool[i % len(view_pool)]
                    s2, _, _ = self._render_plan(fg_view, p2, place_override,
                                                 self.BASE_SEED + 1000 + i, harmonize,
                                                 allow_people=lifestyle)
                    shots[i] = s2

        # 收尾：卸掉裁判階段載入的 Ollama VLM，讓顯卡乾淨交還（下一輪不必再清）。
        self.llm.release_models()
        timings["total"] = round(time.time() - t_total, 2)
        self._notify(progress_cb, "完成", 1.0)
        return StudioResult(
            card=card, plans=plans, shots=shots, copy=copy_pack, timings=timings,
            fg=fg, previews=previews, placement=place_override or Placement(),
            views=views, view_pool=view_pool,
            mode=resolved_mode, product_class=product_class,
        )

    def refine(self, result: StudioResult, plan_idx: int, instruction: str,
               placement: Placement | None = None, harmonize: bool | None = None,
               lifestyle: bool = False,
               ) -> StudioResult:
        """多輪修改：LLM 解析指令為參數差分，只重跑該方案；可同時調整主角擺放。"""
        if not 0 <= plan_idx < len(result.plans):
            raise IndexError(f"plan_idx={plan_idx} 超出範圍（共 {len(result.plans)} 個）")
        if result.fg is None:
            raise ValueError("result 缺少去背前景（fg）")
        place = (placement or result.placement).clamped()
        if harmonize is None:
            harmonize = self.quality == "fine" and not lifestyle

        t0 = time.time()
        # 空指令＝只調主角擺放、不改場景（省一次 LLM 呼叫）
        if (instruction or "").strip():
            new_plan = self.llm.parse_edit(result.plans[plan_idx], instruction)
            self.llm.release_models()
        else:
            new_plan = result.plans[plan_idx]
        pool = result.view_pool or [result.fg]

        # 依原結果的模式分流：reshape 結果必須用重塑路徑重跑，不可走鎖定 inpaint
        # （否則戴在身上的成品會被改回平貼檯面，且觸發 reshape↔inpaint 互卸顛簸）。
        if result.mode == "reshape":
            self._ensure_reshape()
            from .reshape import framing_for
            framing = framing_for(result.card, result.product_class)
            shot, _ = self._render_reshape(result.fg, new_plan, framing,
                                           self.BASE_SEED + plan_idx)
            preview = result.fg
        else:
            self._ensure_inpaint()
            fg_view = pool[plan_idx % len(pool)]  # 沿用該方案原本的角度
            shot, preview, _ = self._render_plan(
                fg_view, new_plan, place, self.BASE_SEED + plan_idx, harmonize,
                allow_people=lifestyle,
            )

        plans = list(result.plans); plans[plan_idx] = new_plan
        shots = list(result.shots); shots[plan_idx] = shot
        previews = list(result.previews)
        if plan_idx < len(previews):
            previews[plan_idx] = preview
        timings = dict(result.timings)
        timings[f"refine_{plan_idx + 1}"] = round(time.time() - t0, 2)
        return StudioResult(
            card=result.card, plans=plans, shots=shots, copy=result.copy,
            timings=timings, fg=result.fg, previews=previews, placement=place,
            views=result.views, view_pool=pool,
            mode=result.mode, product_class=result.product_class,
        )
