# SnapStudio — AI 商品攝影棚：系統架構設計（階段二交付物）

> 一張手機隨手拍的商品照進來，一整套可上架的電商素材出去：
> 去背 → AI 場景 → 物理合理的重新打光 → 多平台文案。
> 全程本機推論（RTX 3090），LLM 與 Diffusion 在參數層級互相咬合。

> **⚠️ 現況更新**：本文為**階段二（6/10）凍結的架構設計**，作為當時的介面契約保留；
> 部分細節於後續迭代演進，**最新現況請見 [WORKFLOW_LOG §10](WORKFLOW_LOG.md) 與
> [DEVLOG §18](../DEVLOG.md)**。主要變更：
> - **文字模型**：主力改為**本機 qwen3:32b**；遠端 Big Pickle 降為**預設關**的可選降級端點。
>   VLM 為本機 `qwen2.5vl:32b-ctx8k`（無 OpenRouter）。
> - **雙模式自動路由**（VLM 看圖判 `product_class`）：rigid→**鎖定模式**（compose/groundgen）、
>   wearable/handheld→**重塑模式**（reshape.py，IP-Adapter 重畫戴/握姿態）。
> - **IC-Light**：從「選配」改為 **UI 預設開 + A 護字**（純 CV 偵測標籤區，產品吃場景光、
>   文字/logo 保持銳利，三層 fallback）。
> - **UI**：改為**單頁 premium 自訂主題**（暗色攝影棚 + 暖琥珀），非多分頁。
> - 新增「**背景全由 AI 決定**」選項。

## 1. 課程技術要求對應

| 作業要求 | 本專題的具體實現 |
|---|---|
| LLM：Prompt Engineering | 場景企劃將口語需求展開為 SDXL prompt + 負面詞 |
| LLM：API 整合 / 本機推論 | 文字企劃/文案與 VLM 識別皆走**本機 Ollama**（qwen3:32b + qwen2.5vl:32b），OpenAI 相容；遠端端點（Big Pickle）可選、預設關（*迭代後更新，原凍結設計為遠端主力*） |
| Diffusion：客製化 Pipeline | **IC-Light 重打光管線在 diffusers 0.39 上重新實作**（UNet conv_in 4→8/12 通道、權重 offset 合併、forward 攔截）— 官方碼僅支援 0.27，本專題為新版改寫 |
| Diffusion：推論加速 | LCM-LoRA 掛載（25 步 → 4-8 步）供「快速預覽 vs 精修輸出」雙檔位 |
| 互動 UI | Gradio 6.17 單頁 premium 自訂主題（暗色攝影棚；*原凍結設計為多分頁*） |

## 2. 系統架構與資料流（v2：inpaint-grounded）

> **v2 重設計動機**：v1「先生成完整背景再把產品貼上去重打光」會讓產品像貼紙
> 浮在場景上、不接地。v2 改為**鎖住產品像素、用 SDXL inpaint 在產品「周圍」
> 生成場景**——接地陰影、反光在同一次去噪自然長出，產品最後原樣貼回。

```
 使用者照片（手機隨手拍）
      │
      ├─────────────────────────────┐
      ▼                             ▼
 ① 去背 matting.py             ② VLM 商品識別 llm.py
   rembg / BiRefNet               「藍色運動飲料鋁罐」
   → RGBA 前景                     → 商品卡 JSON（可由手動描述覆寫）
      │                             │
      │             ┌───────────────┘
      │             ▼
      │      ③ LLM 場景企劃 llm.py（核心咬合點）
      │         商品卡 + 口語需求 → N 組方案 JSON
      │         scene_prompt 描述「檯面＋乾淨背景」（非完整場景）
      │             │
      ▼             ▼
 ④ 主角定位＋雜訊底 compose.py
    依 Placement（大小/水平/垂直/旋轉）把產品擺到畫布
    待生成區填**高頻雜訊**（非純色，否則 inpaint 輸出被錨定成灰）
    → init 圖 + 遮罩（產品黑=保留、周圍白=生成）
      │
      ▼
 ⑤ Inpaint-grounded 生成 groundgen.py（核心咬合點）
    SDXL 9 通道 inpaint 依 scene_prompt 在產品周圍生成場景
    → 接地陰影/反光自然生成 → 產品像素原樣貼回（電商鐵則）
      │
      ▼
 ⑥（選配）IC-Light 光線融合 relight.py（fbc，本專題自製 pipeline）
    讓產品表面光照與場景一致；light_desc 來自 ③；精緻檔預設開
      │
      ▼
 ⑦ LLM 文案生成 llm.py → 蝦皮標題 / 五點賣點 / IG 貼文
      │
      ▼
 輸出素材包：N 張整合成品 + 文案組（Gradio UI，可調擺放／多輪修改）
```

**LLM 與 Diffusion 的咬合點**（避免「兩邊各做各的」）：
- 場景企劃的 `scene_prompt` → 直接餵 SDXL inpaint（⑤）
- 場景企劃的 `light_direction` / `light_desc` → 餵 IC-Light 光線融合條件（⑥）
- 多輪修改：「光再暖一點、背景換大理石」→ LLM 解析為參數差分；
  「移動／縮放／旋轉主角」→ 改 Placement（空指令時免 LLM，僅約 3.5s）→ 只重跑 ⑤⑥

## 3. 模組拆解

| 模組 | 職責 | 輸入 → 輸出 | 技術 | 狀態 |
|---|---|---|---|---|
| `matting.py` | 去背 | RGB → RGBA | rembg（BiRefNet，u2net 備援） | 已實作 |
| `compose.py` | 主角定位＋雜訊底＋遮罩＋貼回 | RGBA + Placement → init/mask/product | PIL/numpy；位置/角度/大小可控 | **v2 新增**，已驗證 |
| `groundgen.py` | inpaint-grounded 場景生成 | init+mask+prompt → 整合成品 | RealVisXL V4 9 通道 inpaint（退路官方 SDXL inpaint；+選配 LCM-LoRA） | **v2 新增/主流程**：1024²、32 步約 6s；allow_people 控人物 |
| `relight.py` | IC-Light 光線融合（選配） | 前景+背景+prompt → 協調光成品 | SD1.5 + IC-Light offset，自製 pipeline | 已實作；降為選配 harmonize 層 |
| `scene.py` | （v1 舊）整張背景生成 | scene_prompt → 背景圖 | RealVisXL V5.0 | 保留供參考，主流程已不用 |
| `llm.py` | VLM 識別／場景企劃／文案／修改解析 | 圖+文 → 結構化 JSON | OpenAI 相容 client；pydantic 驗證 + 重試 + 簡體字防線 | 已實作 |
| `pipeline.py` | 編排器（v2 inpaint-grounded） | — | 單卡 VRAM 調度；Placement/harmonize 參數 | 已實作 |
| `app.py` | Gradio 6 UI（含主角擺放滑桿） | — | gr.Blocks 多分頁 | 已實作 |

## 4. LLM 介面契約（JSON Schema）

### 4.1 商品卡（VLM 商品識別輸出）
```json
{
  "category": "皮夾/錢包",
  "name_guess": "手工棕色皮革短夾",
  "material": "磨砂牛皮",
  "color": "深棕",
  "condition": "二手，狀況良好，輕微使用痕跡",
  "selling_points": ["手工縫線", "真皮質感", "復古色澤"],
  "target_audience": "25-40 歲注重質感的男性"
}
```

### 4.2 場景方案（LLM 場景企劃輸出，一次 N 組）
```json
{
  "plan_name": "職人咖啡館",
  "scene_prompt": "dark walnut wood table, blurred cafe interior, morning ambiance, shallow depth of field, professional product photography",
  "negative_prompt": "cluttered, plastic, text, watermark",
  "light_direction": "left",
  "light_desc": "warm morning window light from left side, soft shadows",
  "mood": "溫暖質感",
  "composition_tip": "商品置於右三分線，左側留白"
}
```
`light_direction ∈ {left, right, top, bottom, front, back}`；`light_desc` 併入 relight prompt。

### 4.3 文案包（LLM 文案輸出）
```json
{
  "shopee_title": "【手工真皮】復古棕色牛皮短夾 二手九成新 質感男夾",
  "bullet_points": ["…五點賣點…"],
  "ig_caption": "…含 emoji 的貼文…",
  "hashtags": ["#真皮錢包", "#手工皮件"]
}
```

所有 LLM 呼叫：強制 JSON 輸出 → pydantic 驗證 → 失敗重試 2 次 → 仍失敗則套用內建預設模板（系統永不因 LLM 掛掉而卡死）。

## 5. 模型與權重清單

| 模型 | 位置 | 大小 | VRAM | 授權 |
|---|---|---|---|---|
| SD1.5 base（fp16 variant） | `weights/sd15/` | 2.1GB | 3.6GB（含 IC-Light） | CreativeML OpenRAIL-M |
| IC-Light v1 fc / fbc | `weights/iclight/` | 1.7GB ×2 | 同上 | 碼 Apache-2.0 / 權重 OpenRAIL-M |
| RealVisXL V5.0（場景生成主力） | `weights/realvisxl/` | 6.9GB | 9.6GB | OpenRAIL++ |
| SDXL base 1.0（借設定檔；RealVisXL 缺檔退路） | HF 快取（已存在） | 27GB | 9.6GB | OpenRAIL++ |
| LCM-LoRA SD1.5 | `weights/lcm-lora-sdv1-5/` | 135MB | +0 | OpenRAIL-M |
| rembg（BiRefNet/u2net） | 自動下載（GitHub，不經 HF CDN） | ~180MB | CPU | MIT/Apache |
| LLM 文字（主力）：本機 `qwen3:32b`（Ollama, q4） | 本機 | ~20GB | LLM 階段獨佔 | Apache-2.0 |
| LLM 文字（可選降級，預設關）：遠端 Big Pickle（opencode zen，實測路由到 deepseek 系推理模型） | API | — | — | 免費端點 |
| VLM：本機 `qwen2.5vl:32b-ctx8k`（Ollama, q4） | 本機 | ~21GB | LLM 階段獨佔 | Apache-2.0 |

> 環境鐵則（已實測）：本機到 HF CDN 長連線會停滯 → 權重一律 `wget -c` 預先下載，
> 執行一律 `HF_HUB_OFFLINE=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 6. 已驗證的技術事實（POC 數據，RTX 3090）

- IC-Light fc 模式在 diffusers 0.39.0.dev0 改寫成功：**768×768、25 步、3.1 秒/張、VRAM 峰值 3.6GB**（`poc/poc_iclight.py`）
- 商品主體保留度：皮革紋理/縫線完整，光向與色溫確實改變（佐證見 `poc/poc_iclight.py`；IC-Light A 護字 before/after 見 `examples/showcase/iclight_ab.png`）
- SDXL 場景生成：1024²、30 步、約 8-11 秒/張、VRAM 9.6GB（前期已驗證）
- 已知改進點：u2net 會把鄰近物切進前景 → 換 BiRefNet；fc 背景僅文字近似 → 主線用 fbc
- 型別陷阱：新建 conv_in 預設 fp32，必須 `.to(unet.dtype)`（已踩過）

## 7. 里程碑（截止 6/23）

| 日期 | 里程碑 | 驗收標準 |
|---|---|---|
| 6/10 ✅ | M1 POC：IC-Light 改寫跑通 | 兩張重打光成品 |
| 6/11 | M2 模組化：matting + relight（含 fbc） | fbc 指定背景圖融合成功 |
| 6/12 | M3 llm.py + scene.py | 商品卡/場景方案 JSON 穩定產出 |
| 6/13 | M4 pipeline 整合 | CLI 一鍵：照片 → 素材包 |
| 6/14-15 | M5 Gradio UI | 上傳→方案選擇→成品→文案 全流程可互動 |
| 6/16-17 | M6 打磨 + 緩衝 | LCM 快速檔、多輪修改、真實照片測試 |
| 6/18-19 | M7 README + Workflow Log + demo 錄影 | 三份交付物完稿 |
| 6/20-23 | 緩衝；GitHub push（等使用者確認） | — |

## 8. 風險與降級鏈

| 風險 | 等級 | 降級方案 |
|---|---|---|
| fbc 融合品質不穩（前景邊緣、透視不合） | 中 | 退 fc 模式（已驗證）+ 場景描述寫進 prompt；或 fbc 失敗品自動重抽 seed |
| VLM 不可用 / 識別錯誤 | 低 | UI 提供手動商品描述欄，商品卡可編輯後再企劃 |
| LLM JSON 格式飄移 | 低 | pydantic 驗證 + 重試 + 預設模板 |
| gradio 6 API 與舊範例不相容 | 中 | 開發時以官方 migration guide 為準；UI 只用 Blocks/Image/Gallery/Textbox 等穩定元件 |
| VRAM 峰值疊加（SDXL 級場景模型 + SD1.5 同駐） | 低 | pipeline.py 統一調度：場景模型用完 offload 到 CPU 再載 relight（24GB 實際可同駐，保險起見仍做調度） |
