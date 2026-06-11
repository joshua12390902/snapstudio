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
                manual_desc, session, progress=gr.Progress()):
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
# 版面（gradio 6：theme/css 移到 launch()，此處只建結構）
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SnapStudio — AI 商品攝影棚") as demo:
        gr.Markdown("# SnapStudio — AI 商品攝影棚")
        session_state = gr.State(None)

        with gr.Tab("一鍵攝影棚"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Gallery(
                        label="商品照片（可拖多張：第一張為主圖，其餘當不同角度）",
                        type="pil", interactive=True, columns=4, height=300,
                        object_fit="contain",
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
                    generate_btn = gr.Button("開始生成", variant="primary")
                with gr.Column(scale=2):
                    mode_status = gr.Markdown("")
                    card_json = gr.JSON(label="商品卡（VLM 自動識別）")
                    manual_tb = gr.Textbox(
                        label="手動商品描述（選填，覆寫自動識別）",
                        placeholder="例：手工棕色皮革短夾，磨砂牛皮，二手九成新",
                        lines=2,
                    )
                    gallery = gr.Gallery(
                        label="成品方案（點選可切換下方文案）",
                        columns=2, height=420, object_fit="contain",
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
                    harmonize_chk, lifestyle_chk, manual_tb, session_state],
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
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1400px !important; margin: auto;}",
        footer_links=["gradio"],
        show_error=True,
    )
