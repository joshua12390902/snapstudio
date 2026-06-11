# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SnapStudio — AI 商品攝影棚（深度生成模型課程期末專題）。一張手機商品照 →
去背 → VLM 識別 → LLM 場景企劃 → Diffusion 生成 → 多平台文案，全程本機推論
（單張 RTX 3090 24GB）。設計賣點：LLM 與 Diffusion 在**參數層級互相咬合**。

## 環境與執行

- **Python 直譯器在 repo 外**：`/workspace/.venv-1/bin/python`（Python 3.10，GPU 全套）。
  repo 內沒有 `.venv`（README 的建立步驟未實際執行）。
- `snapstudio/config.py` 匯入時自動 `setdefault` `HF_HUB_OFFLINE=1` 與
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，所以 import 任何 snapstudio
  模組就已離線。從 repo 根目錄外跑 script 需自帶 `PYTHONPATH=.`。

```bash
cd /workspace/Deep_Generative_Model/HW7_snapstudio
# Gradio UI（http://localhost:7860）
/workspace/.venv-1/bin/python app.py
# CLI 一鍵素材包
PYTHONPATH=. /workspace/.venv-1/bin/python cli.py --image <img> --brief "文青風" --n 3
# 批次生成（讀 recipe JSON，GPU 序列、安全）
PYTHONPATH=. /workspace/.venv-1/bin/python examples/opt2/gen_round.py <recipe.json>
# 煙霧驗證（無正式測試框架）：import 不報錯＝結構 OK
/workspace/.venv-1/bin/python -c "import app"
```

- **GPU script 一律背景跑**（載模型+生成數十秒～數分鐘）：
  `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. /workspace/.venv-1/bin/python <script>`。
- **沒有 pytest / ruff / flake8 設定**，也沒有 `tests/`。改動後用上面的 import 煙霧測，
  GPU 路徑則跑一張小圖目視。

## 架構大圖（需跨檔才看得懂的部分）

編排器 `snapstudio/pipeline.py::SnapStudio.process()` 是主軸，串起一條
**「LLM 決策 → Diffusion 執行」**的鏈，並做 VRAM 調度：

1. **去背** `matting.py`（rembg bbox → SAM2 box refine，0% 白暈邊）。
2. **VLM 識別** `llm.py::identify_product` → `ProductCard`，其中
   `product_class`(rigid/wearable/handheld) 與 `worn_framing` 由 VLM 看圖決定。
3. **AI 自動路由模式**：rigid→**鎖定模式**、wearable/handheld→**重塑模式**
   （`mode="auto"`，可手動 `locked`/`reshape` 覆蓋）。VLM 離線時才用
   `fallback_product_class` 的關鍵字網補判——**關鍵字只是降級備援，主判給 LLM**。
4. **場景企劃** `llm.py::plan_scenes(mode=...)` → N 組 `ScenePlan`（英文 scene_prompt/
   光向/擺位）。
5. **生成**：依模式走兩條互斥路徑（見下）。`release_models()` 在 LLM 階段後卸掉
   Ollama，**diffusion 才載入** → LLM 與 diffusion 不同時佔 VRAM，各自可用滿 24GB。
6. **文案** `llm.py::write_copy` → `CopyPack`（蝦皮標題/賣點/IG）。

**LLM↔Diffusion 咬合點**：`scene_prompt` 直接餵 SDXL inpaint、`light_direction`
餵程式接地陰影與 IC-Light、`worn_framing` 餵重塑取景。

### 渲染雙模式（互斥，單卡一次只駐一個 diffusion 模型）

- **鎖定模式（rigid）** `compose.py` + `groundgen.py`：
  `build_scene_inputs` 用**高頻雜訊底**（非純色，純色會把 inpaint 錨成同色）+ 遮罩鎖住
  產品像素；`SceneInpainter`（RealVisXL 9 通道 inpaint）在產品**周圍**生成場景；
  `paste_back` 把原始產品像素貼回並加**程式三層接地陰影**（接地靠程式，不靠 prompt）。
  多角度：`process(angle_images=[...])` 把多張照組成 `view_pool`，各方案輪流取角度。
- **重塑模式（wearable/handheld）** `reshape.py::ReshapeStudio`：
  RealVisXL text2img + IP-Adapter（`weights/ip-adapter/`，plus-vit-h）保留產品**身份**、
  讓模型重畫姿態（戴手腕/手持）；取景片語 `framing_for` 取自 VLM 的 `worn_framing`。
  `composite_real_face` 用色彩自適應偵測把**真實平面細節面**（錶盤/標籤）合成回去補小細節。

`_ensure_inpaint` / `_ensure_reshape` / `_ensure_relight` 三者互相 `unload`，保證
單卡一次只駐一個 SDXL/SD pipeline。

### LLM 契約鏈（永不因 LLM 掛掉而崩）

`llm.py::LLMClient._structured_call`：端點降級鏈（Big Pickle 遠端 → 本機 Ollama）→
強制 JSON → pydantic 驗證（`schemas.py`）→ 重試 → 仍失敗套 `DEFAULT_*` 模板。
**對外永不拋例外**（`identify_product` 失敗回 None，由 pipeline 退 `_fallback_card`）。
輸出含**簡體字防線**（`_SIMP_FIXTABLE` 安全轉正體、`_SIMP_DETECT` 命中重試）——本 repo
一律繁體中文，勿引入簡體字。

## 環境鐵則（踩過的坑）

- **HF CDN 在本機會停滯**：勿用 `huggingface_hub` 直下大檔。權重用
  `scripts/download_weights.sh`（wget `-c` 斷點續傳）或臨時 `curl -C -`，下完離線載入。
- **無 nvcc、無 EGL**：`tripo3d.py` 用純 torch NeRF 算 3D（torchmcubes 已 stub）。
- **RealVisXL 單檔載入需 SDXL 設定檔快取**（`stabilityai/stable-diffusion-xl-base-1.0`、
  `madebyollin/sdxl-vae-fp16-fix`）；首次缺檔可暫 `HF_HUB_OFFLINE=0` 抓設定檔（數 MB）。
- **權重路徑全在 `config.py`**（`REALVISXL_PATH`、`INPAINT_DIR`、`weights/ip-adapter/`…）。
  `weights/`（約 36GB）、`examples/_archive/`、`examples/products/raw/` 已 gitignore。
- **版本敏感**：`diffusers==0.39.0.dev0`（git pin，IC-Light/重塑所需）、`transformers==5.3.0`、
  `gradio==6.17.3`。IC-Light（`relight.py`）是在 0.39 重新實作（conv_in 4→8/12 通道）。

## Ollama 模型

VLM `qwen2.5vl:32b`、文字本地備援 `qwen3:32b`（皆 q4，~20GB，靠 LLM 階段獨佔 VRAM）；
文字主力為遠端 Big Pickle。皆 OpenAI 相容，`SNAPSTUDIO_*` 環境變數可覆寫（見 `config.py`）。

## 進行中

`poc/worn_harmonize_poc.py` + `snapstudio/pipelines/diff_img2img.py`（Differential
Diffusion 社群 pipeline）：重塑手錶的「貼真實錶頭→軟凍結錶盤→接縫和諧化」混合法 POC，
**尚未整進主 pipeline**。差分擴散 `map` 語意：**1=凍結保真、0=自由重繪**（易記反）。
