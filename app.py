"""SnapStudio Gradio 介面：一鍵攝影棚（單 Tab）。

依賴 snapstudio.pipeline.SnapStudio（lazy 載入；介面以 pipeline.py 為準）：

    studio = SnapStudio(quality)                # 模型常駐單例；set_quality() 可切檔
    result = studio.process(
        image, user_brief="", n_plans=3,
        progress_cb=None,                       # callable(stage: str, frac: float)
        manual_desc=None,                       # str|None 手動商品描述（覆寫 VLM）
        mode="auto",                            # auto/locked/reshape 渲染模式路由
        angle_images=None,                      # {"three_quarter": PIL, "top": PIL}
    ) -> StudioResult(card, plans, shots, copy, timings, fg, mode, product_class)

studio.refine(result, plan_idx, instruction) 仍是可用的後端能力（依 result.mode
分流 locked/reshape），但目前 UI 未綁定微調工作台（已移除），僅供程式呼叫。

UI 的 session state 結構（gr.State 內存 Python 物件）：
    {"result": StudioResult,                    # 含 fg/views，refine 重跑用
     "product_card": dict,
     "plans": [{"plan": dict, "image": PIL.Image, "copy": dict}, ...]}
文案包依規格只對第一方案生成一次，UI 各方案共用同一份。
"""
import sys
import traceback
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snapstudio import config  # noqa: E402  # 設定 HF_HUB_OFFLINE 等環境變數

EXAMPLE_IMAGE = config.EXAMPLES / "test_product_input.png"
QUALITY_MAP = {"快速": "fast", "精緻": "fine"}

# ---------------------------------------------------------------------------
# pipeline lazy 全域單例：避免 import app 時就載入大模型，
# 也讓 pipeline.py 尚未完成時 UI 仍能啟動展示。
# ---------------------------------------------------------------------------
_studio = None


def _get_studio(quality: str = "fine"):
    global _studio
    if _studio is None:
        from snapstudio.pipeline import SnapStudio
        _studio = SnapStudio(quality=quality)
    else:
        _studio.set_quality(quality)
    return _studio


def _session_from_result(result) -> dict:
    """StudioResult → UI session dict（文案各方案共用同一份）。"""
    copy = result.copy.model_dump() if result.copy is not None else {}
    return {
        "result": result,
        "product_card": result.card.model_dump(),
        "plans": [
            {"plan": plan.model_dump(), "image": shot, "copy": copy}
            for plan, shot in zip(result.plans, result.shots)
        ],
    }


# ---------------------------------------------------------------------------
# 顯示用輔助
# ---------------------------------------------------------------------------
def _copy_text(copy: dict) -> str:
    """把文案包 JSON 排成可直接複製的多行文字。"""
    if not isinstance(copy, dict):
        return str(copy or "")
    lines = []
    if copy.get("shopee_title"):
        lines += ["【蝦皮標題】", copy["shopee_title"], ""]
    if copy.get("bullet_points"):
        lines += ["【賣點】"] + [f"・{p}" for p in copy["bullet_points"]] + [""]
    if copy.get("ig_caption"):
        lines += ["【IG 貼文】", copy["ig_caption"], ""]
    if copy.get("hashtags"):
        lines += ["【Hashtags】", " ".join(copy["hashtags"])]
    return "\n".join(lines).strip()


def _plan_label(i: int, item: dict) -> str:
    name = (item.get("plan") or {}).get("plan_name", "未命名")
    return f"方案 {i + 1}：{name}"


def _gallery_items(session: dict) -> list:
    return [(it["image"], _plan_label(i, it))
            for i, it in enumerate(session["plans"])]


def _skip(n: int) -> tuple:
    return tuple(gr.skip() for _ in range(n))


def _warn(msg: str):
    gr.Warning(msg)


# ---------------------------------------------------------------------------
# 事件處理
# ---------------------------------------------------------------------------
N_GEN_OUTPUTS = 5  # mode_status, card, gallery, copy, state

_CLASS_LABEL = {"rigid": "剛性擺台", "wearable": "穿戴", "handheld": "手持"}
_MODE_LABEL = {"locked": "鎖定模式（像素精準）", "reshape": "重塑模式（戴上身/手持）"}


def _mode_status_md(result, requested_mode: str) -> str:
    cls = _CLASS_LABEL.get(getattr(result, "product_class", "rigid"), "剛性擺台")
    md = _MODE_LABEL.get(getattr(result, "mode", "locked"), "鎖定模式")
    how = "AI 自動判斷" if requested_mode == "auto" else "手動指定"
    return f"**🤖 {how}**：產品類別＝**{cls}** → 採用 **{md}**"


def _gallery_to_pils(value) -> list:
    """gr.Gallery 輸入值 → PIL list。元素可能是 (img, caption) tuple、路徑字串、
    PIL Image 或 numpy 陣列，盡量穩健地轉成 RGB PIL。"""
    pics = []
    for item in (value or []):
        if isinstance(item, (tuple, list)):
            item = item[0] if item else None
        if item is None:
            continue
        if isinstance(item, str):
            try:
                item = Image.open(item)
            except Exception:  # noqa: BLE001
                continue
        elif not hasattr(item, "convert"):  # numpy 陣列等
            try:
                item = Image.fromarray(item)
            except Exception:  # noqa: BLE001
                continue
        pics.append(item.convert("RGB"))
    return pics


def on_generate(images, brief, num_plans, mode, harmonize, lifestyle,
                bg_ai_free, manual_desc, session, progress=gr.Progress()):
    """Tab1 一鍵生成：多張照片 → 商品卡 → N 組方案成品 + 文案（固定精緻檔）。

    images：gr.Gallery 多圖上傳。第一張為主圖（識別＋front），其餘當角度池。
    """
    pics = _gallery_to_pils(images)
    if not pics:
        _warn("請先上傳商品照片")
        return _skip(N_GEN_OUTPUTS)
    primary, extras = pics[0], pics[1:]
    try:
        progress(0.0, desc="載入管線…")
        studio = _get_studio("fine")
        result = studio.process(
            primary,
            user_brief=(brief or "").strip(),
            n_plans=int(num_plans),
            manual_desc=(manual_desc or "").strip() or None,
            harmonize=bool(harmonize),
            lifestyle=bool(lifestyle),
            bg_ai_free=bool(bg_ai_free),
            angle_images=extras or None,
            mode=mode or "auto",
            progress_cb=lambda stage, frac: progress(frac, desc=stage),
        )
        if not result.plans:
            raise RuntimeError("pipeline 未回傳任何方案")
        session = _session_from_result(result)
        first = session["plans"][0]
        return (
            _mode_status_md(result, mode or "auto"),
            session["product_card"],
            _gallery_items(session),
            _copy_text(first.get("copy", {})),
            session,
        )
    except Exception as exc:  # noqa: BLE001  # UI 不可崩潰，一律轉成 Warning
        traceback.print_exc()
        _warn(f"生成失敗：{exc}")
        return _skip(N_GEN_OUTPUTS)


def on_gallery_select(session, evt: gr.SelectData):
    """點選成品 → 切換右側文案到該方案。"""
    if not session or evt.index is None:
        return gr.skip()
    try:
        return _copy_text(session["plans"][evt.index].get("copy", {}))
    except (IndexError, TypeError):
        return gr.skip()


def _btn_off():
    return gr.Button(interactive=False)


def _btn_on():
    return gr.Button(interactive=True)


# ---------------------------------------------------------------------------
# 視覺主題：沉靜黑色攝影棚 + 一束暖琥珀（研究 Linear/Vercel/Stripe/攝影棚站而定）
# ---------------------------------------------------------------------------
_AMBER = gr.themes.Color(
    c50="#FCF3E3", c100="#FAE7C7", c200="#F6D69D", c300="#F3C97E",
    c400="#F1C06E", c500="#F4B860", c600="#D89A45", c700="#A9762F",
    c800="#7A5320", c900="#4D3413", c950="#2A1C0A",
)
_INK = gr.themes.Color(
    c50="#EDEDED", c100="#D6D6D8", c200="#B6B6BA", c300="#8A8A8F",
    c400="#6A6A6F", c500="#4A4A4E", c600="#3A3A3D", c700="#262629",
    c800="#161618", c900="#0F0F10", c950="#0A0A0B",
)


class SnapStudioTheme(gr.themes.Base):
    def __init__(self) -> None:
        super().__init__(
            primary_hue=_AMBER, secondary_hue=_AMBER, neutral_hue=_INK,
            text_size=gr.themes.sizes.text_md,
            spacing_size=gr.themes.sizes.spacing_md,
            radius_size=gr.themes.sizes.radius_md,
            # 字體不走 theme.font（混用 GoogleFont+字串會觸發 gradio 主題比較的 __eq__ 崩潰），
            # 改由 launch(head=SNAP_HEAD) 注入 <link> + CSS 變數 --font 設定，見下。
        )
        super().set(
            body_background_fill="#0A0A0B", body_background_fill_dark="#0A0A0B",
            body_text_color="#EDEDED", body_text_color_dark="#EDEDED",
            body_text_color_subdued="#8A8A8F", body_text_color_subdued_dark="#8A8A8F",
            background_fill_primary="#0A0A0B", background_fill_primary_dark="#0A0A0B",
            background_fill_secondary="#161618", background_fill_secondary_dark="#161618",
            color_accent="#F4B860", color_accent_soft="#262629",
            border_color_primary="#262629", border_color_primary_dark="#262629",
            border_color_accent="#F4B860",
            link_text_color="#F4B860", link_text_color_hover="#F1C06E",
            link_text_color_dark="#F4B860", link_text_color_hover_dark="#F1C06E",
            block_background_fill="#161618", block_background_fill_dark="#161618",
            block_border_width="1px", block_border_color="#262629",
            block_border_color_dark="#262629", block_radius="12px", block_padding="20px",
            block_shadow="inset 0 0 0 1px rgba(255,255,255,0.05)",
            block_shadow_dark="inset 0 0 0 1px rgba(255,255,255,0.05)",
            block_label_background_fill="#161618", block_label_text_color="#8A8A8F",
            block_label_text_weight="500", block_label_radius="8px",
            block_title_text_color="#EDEDED", block_title_text_weight="600",
            block_title_text_size="14px",
            block_info_text_color="#8A8A8F", block_info_text_size="12px",
            layout_gap="16px", panel_background_fill="#161618", panel_border_width="1px",
            container_radius="16px",
            input_background_fill="#0F0F10", input_background_fill_dark="#0F0F10",
            input_background_fill_focus="#161618",
            input_border_color="#262629", input_border_color_focus="#F4B860",
            input_border_color_hover="#3A3A3D", input_border_width="1px",
            input_radius="8px", input_shadow="none",
            input_shadow_focus="0 0 0 2px rgba(244,184,96,0.40)",
            input_placeholder_color="#6A6A6F", input_text_size="14px",
            button_primary_background_fill="#F4B860",
            button_primary_background_fill_hover="#F1C06E",
            button_primary_background_fill_dark="#F4B860",
            button_primary_background_fill_hover_dark="#F1C06E",
            button_primary_text_color="#0A0A0B", button_primary_text_color_hover="#0A0A0B",
            button_primary_border_color="#F4B860",
            button_primary_shadow="0 0 0 1px rgba(244,184,96,0.40), 0 8px 24px rgba(244,184,96,0.15)",
            button_primary_shadow_hover="0 0 0 1px rgba(244,184,96,0.55), 0 10px 28px rgba(244,184,96,0.22)",
            button_secondary_background_fill="#161618",
            button_secondary_background_fill_hover="#1E1E21",
            button_secondary_text_color="#EDEDED",
            button_secondary_border_color="#262629",
            button_secondary_border_color_hover="#3A3A3D",
            button_large_radius="8px", button_large_padding="10px 16px",
            button_large_text_size="14px", button_large_text_weight="600",
            button_border_width="1px",
            button_transition="all 0.18s cubic-bezier(0.4,0,0.2,1)",
            button_transform_hover="translateY(-1px)", button_transform_active="translateY(0px)",
            slider_color="#F4B860", loader_color="#F4B860",
            error_background_fill="#2A1414", error_border_color="#7A2E2E",
            error_text_color="#F0A0A0", code_background_fill="#0F0F10",
            prose_text_size="16px", prose_header_text_weight="600",
            shadow_drop="0 1px 2px rgba(0,0,0,0.4)", shadow_drop_lg="0 8px 24px rgba(0,0,0,0.45)",
        )


SNAP_THEME = SnapStudioTheme()

SNAP_CSS = """
.gradio-container {
  max-width: 1280px !important; margin: auto !important;
  padding: 28px 24px 64px !important;
  --font: 'Inter', ui-sans-serif, system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  background:
    radial-gradient(1100px 520px at 50% -8%, rgba(244,184,96,0.055), transparent 62%),
    #0A0A0B !important;
}
.snap-hero-title, .prose h1, .prose h2, .prose h3, .section-header {
  font-family: 'Space Grotesk', ui-sans-serif, sans-serif !important;
}
#snap-hero { margin: 4px 2px 26px; border: none !important; background: transparent !important; }
.snap-badge {
  display: inline-block; font-size: 12px; letter-spacing: .04em; color: #F4B860;
  border: 1px solid rgba(244,184,96,0.35); border-radius: 999px;
  padding: 5px 13px; margin-bottom: 18px; background: rgba(244,184,96,0.06);
}
.snap-hero-title {
  font-family: 'Space Grotesk', ui-sans-serif, sans-serif;
  font-size: 40px; line-height: 1.1; font-weight: 600; letter-spacing: -0.02em;
  margin: 0 0 10px; color: #EDEDED;
}
.snap-hero-title .accent { color: #F4B860; }
.snap-hero-sub { font-size: 16px; line-height: 1.6; color: #8A8A8F; margin: 0; max-width: 640px; }
.snap-card {
  background: #161618 !important; border: 1px solid #262629 !important;
  border-radius: 16px !important; padding: 22px !important;
  animation: snapfade .4s cubic-bezier(.4,0,.2,1) both;
}
@keyframes snapfade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
#snap-upload { border-radius: 12px !important; }
#snap-generate-btn { width: 100% !important; margin-top: 6px; }
#snap-gallery img {
  transition: transform .25s cubic-bezier(.4,0,.2,1), filter .25s; border-radius: 10px;
}
#snap-gallery img:hover { transform: scale(1.015); filter: brightness(1.06); }
#snap-mode-status p {
  display: inline-block; font-size: 13px; color: #F4B860; margin: 0;
  background: rgba(244,184,96,0.08); border: 1px solid rgba(244,184,96,0.25);
  border-radius: 999px; padding: 4px 12px;
}
footer { opacity: .5; }
"""

# Google Fonts 經 <head> 注入（繞過 theme.font 的 __eq__ 崩潰）；外網不可達時瀏覽器自動退系統字
SNAP_HEAD = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500&display=swap'>"
)


# ---------------------------------------------------------------------------
# 版面（gradio 6：theme/css 移到 launch()，此處只建結構）
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SnapStudio — AI 商品攝影棚") as demo:
        gr.HTML(
            "<span class='snap-badge'>本機推論 · 單張 RTX 3090</span>"
            "<h1 class='snap-hero-title'>Snap<span class='accent'>Studio</span>"
            " — AI 商品攝影棚</h1>"
            "<p class='snap-hero-sub'>一張手機商品照，自動去背、識別、企劃打光，"
            "生成電商大片與多平台文案。</p>",
            elem_id="snap-hero",
        )
        session_state = gr.State(None)

        with gr.Tab("一鍵攝影棚"):
            with gr.Row():
                with gr.Column(scale=1, elem_id="snap-controls",
                               elem_classes=["snap-card"]):
                    img_in = gr.Gallery(
                        label="商品照片（可拖多張：第一張為主圖，其餘當不同角度）",
                        type="pil", interactive=True, columns=4, height=300,
                        object_fit="contain", elem_id="snap-upload",
                    )
                    gr.Markdown(
                        "丟同一產品的多張角度照（正面＋3/4 側＋俯視…），系統自動讓"
                        "不同方案輪流用不同角度；只丟一張也行。真實照比 AI 重建銳利。"
                    )
                    brief_tb = gr.Textbox(
                        label="需求描述",
                        placeholder="質感文青風、目標客群上班族",
                        lines=2,
                    )
                    nplans_slider = gr.Slider(
                        minimum=1, maximum=4, step=1, value=2, label="方案數",
                    )
                    mode_radio = gr.Radio(
                        choices=[("自動判斷（AI 決定）", "auto"),
                                 ("鎖定：剛性擺台", "locked"),
                                 ("重塑：戴上身/手持", "reshape")],
                        value="auto", label="渲染模式",
                        info="自動＝由 VLM 看圖判斷：香水/罐等剛性→鎖定（像素精準）；"
                             "手錶/戒指/手把等→重塑（戴在身上/握在手中，姿態可重畫）",
                    )
                    lifestyle_chk = gr.Checkbox(
                        value=False, label="情境生活照（含人物／實驗性）",
                        info="⚠ 實驗性：產品鎖在最上層，生成的人體可能被產品遮擋而失真。"
                             "主打請用預設＝乾淨電商商品背景（無人物，最穩、最像廣告大片）",
                    )
                    harmonize_chk = gr.Checkbox(
                        value=False, label="AI 光線融合（IC-Light）",
                        info="讓商品表面光照與場景一致；較精緻但多約 2s/張",
                    )
                    bg_ai_chk = gr.Checkbox(
                        value=False, label="背景全由 AI 決定",
                        info="忽略上方需求描述的風格，讓 AI 以創意總監身分自選最適合此商品的"
                             "多樣高質感背景；仍守住產品主角／接地／單光源／道具不擋產品",
                    )
                    generate_btn = gr.Button("開始生成", variant="primary",
                                             size="lg", elem_id="snap-generate-btn")
                with gr.Column(scale=2, elem_id="snap-outputs",
                               elem_classes=["snap-card"]):
                    mode_status = gr.Markdown("", elem_id="snap-mode-status")
                    card_json = gr.JSON(label="商品卡（VLM 自動識別）")
                    manual_tb = gr.Textbox(
                        label="手動商品描述（選填，覆寫自動識別）",
                        placeholder="例：手工棕色皮革短夾，磨砂牛皮，二手九成新",
                        lines=2,
                    )
                    gallery = gr.Gallery(
                        label="成品方案（點選可切換下方文案）",
                        columns=2, height=420, object_fit="contain",
                        elem_id="snap-gallery",
                    )
                    copy_tb = gr.Textbox(
                        label="文案包", lines=12, buttons=["copy"],
                        interactive=False,
                    )

        # --- 事件串接（執行中 disable 按鈕，結束後恢復） ---
        gen_outputs = [mode_status, card_json, gallery, copy_tb, session_state]
        generate_btn.click(
            _btn_off, None, generate_btn, api_visibility="private",
        ).then(
            on_generate,
            inputs=[img_in, brief_tb, nplans_slider, mode_radio,
                    harmonize_chk, lifestyle_chk, bg_ai_chk, manual_tb, session_state],
            outputs=gen_outputs,
        ).then(
            _btn_on, None, generate_btn, api_visibility="private",
        )

        gallery.select(on_gallery_select, inputs=[session_state],
                       outputs=[copy_tb])

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        theme=SNAP_THEME,
        css=SNAP_CSS,
        head=SNAP_HEAD,
        footer_links=["gradio"],
        show_error=True,
    )
