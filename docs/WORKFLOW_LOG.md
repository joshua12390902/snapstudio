# SnapStudio — Agent 協作紀錄（Workflow Log）

> 本文件是課程指定交付物「Agent 協作紀錄」，記錄 SnapStudio（AI 商品攝影棚）
> 從題目發想、方向淘汰、架構凍結到核心實作的完整 Agent driven 開發歷程。
> 所有事件、數據與失敗都有佐證檔案（多數在 `docs/exploration/`，含檔案時間戳），
> 誠實記錄走錯的路與修正過程——那正是這份文件的價值所在。

---

## 1. 工具鏈與協作模式總覽

| 角色 | 工具 / 模型 | 用途 |
|---|---|---|
| 主代理（編排者） | Claude Code（Fable 5） | 拆解任務、發起多代理工作流、與使用者互動（AskUserQuestion） |
| 研究代理 | WebSearch / WebFetch | 查權重是否公開、授權條款、技術趨勢；**結論必須附來源** |
| 查證代理 | Bash + 本機原始碼閱讀 | 對研究報告做「不信任覆核」——直接讀 `/workspace/.venv-1` 已安裝套件的原始碼 |
| 實測代理 | Bash + RTX 3090 | 在本機跑 POC / benchmark，量秒數、FPS、峰值 VRAM，寫進 `results.jsonl` |
| 評審代理 | 三個獨立 lens（教授／工程／demo） | 對提案獨立評分，禁止互相參考 |
| 建造代理 | 8 個平行子代理 | 階段三～四：各模組 / UI / 文件平行實作（本文件即出自其中之一） |
| App 後端 LLM | Big Pickle（GLM-4.6）/ Ollama（qwen3:14b、qwen2.5vl） | 成品 App 內的場景企劃、VLM 商品識別、文案生成 |

貫穿全程的兩條鐵則（都是踩坑後立的，見第 7 節）：

1. **權重下載一律 `wget -c` 斷點續傳、執行一律 `HF_HUB_OFFLINE=1`**——本機到 HF CDN 的長連線會停滯。
2. **研究代理的任何「可直接用」結論，必須由查證代理讀本機原始碼覆核**——攔下過一次研究幻覺。

---

## 2. 階段一：發想與 9 代理評選工作流（6/10 上午）

### 2.1 從 4 個方向到 2 個提案

主代理先依課程要求（LLM prompt engineering + API 整合、Diffusion 客製 pipeline、
推論加速、互動 UI）發散出 **4 個候選方向**，用 `AskUserQuestion` 讓使用者圈選，
使用者保留 2 個進入正式評選。

### 2.2 9 代理評選工作流

針對 2 個提案開出 **6 研究代理 + 3 評審代理**：

- **6 研究代理** = 2 提案 × 3 個維度（技術可行性／時程風險／差異化亮點）。
  每個代理同時做兩件事：WebSearch 查權重與授權是否真的拿得到，
  以及在本機 3090 上做最小實測（模型載得起來嗎、一張圖幾秒）。
- **3 評審代理**以三種互不重疊的 lens 獨立評分（評審 prompt 見第 8 節）：
  - **教授視角**：課程要求覆蓋度、技術深度
  - **工程視角**：截止日（6/23）前做得完嗎、本機資源夠嗎
  - **demo 視角**：現場演示的張力與互動性

評審結果 **2:1**，由「PosePainter：照片 → 骨架 → 同姿勢角色生成」勝出。
但主代理沒有直接開工，而是與使用者約定「**半天 POC 定生死**」：
當天上午把端到端鏈路跑通才算數，跑不通就換方向。事後證明這個決策機制
非常關鍵——POC 揭露的問題（第 3 節）與後續兩次方向迭代（第 4 節），
最後把專題帶到了一個完全不同、但更扎實的題目。

---

## 3. 半天 POC 定生死：PosePainter 驗證實錄

佐證：`docs/exploration/poc_posepainter.py`（10:00）→ `poc_part2.py`（10:05）→
`poc_part3.py`（10:07）→ `poc_part4.py`（10:10）→ `07_grid_v2.png`（10:22）。
四個腳本的演進本身就是除錯紀錄，半天內踩了四個坑（技術細節彙整於第 7 節）：

1. **第一版腳本雙 pipeline 同駐 → CUDA OOM**。
   SDXL base 生來源照片後，ControlNet pipeline 用 `from_pipe` 接上，
   兩套管線殘留同駐爆了 24GB。`poc_part2.py` 改為「只載單一 ControlNet 管線」
   （正式版的記憶體配置），峰值 12.3GB，順利補完。
2. **研究代理報告 DWPose「走 onnx 可直接用」是幻覺**。
   查證代理讀本機 `controlnet_aux` 原始碼，發現 DWPose 實際要拉 mmcv 全家桶
   （mmcv/mmdet/mmpose），在本環境裝不得（會動到 torch）。
   改用 **YOLO11-pose（COCO-17）+ 手寫 COCO17→OpenPose18 關鍵點對映**
   （`poc_part4.py` 的 `C2O` 對照表，neck 用雙肩中點補出），
   再借 `controlnet_aux.dwpose.util.draw_bodypose` 渲染標準骨架圖。
3. **`controlnet_conditioning_scale=0.8` 鎖不住姿勢**。
   高踢腿的腳被「史詩山地戰場」這種場景 prompt 拉走，`poc_part3.py`
   調到 **1.0** 才完全鎖住（對照 `05_grid.png` vs `07_grid_v2.png`）。
4. 端到端跑通：照片 → 骨架 → 騎士／賽博格同姿勢生成，POC 判定「活」。

**結論：POC 過關，但代價也看清了**——姿態鏈每個環節都要查證，
而且這個方向的 demo 價值高度依賴「即時性」，這直接引出下一節。

---

## 4. 方向迭代：即時化實測 → 實用轉向 → 選定商品攝影棚

### 4.1 使用者：「能不能做成即時／影片？」→ 4 代理實測

PosePainter 要好玩就得即時（魔鏡：人在鏡頭前動，角色同步動）。
主代理開 **4 代理工作流**：3 個網路研究代理（少步蒸餾方案：LCM-LoRA、
Hyper-SD、TCD 的論文與模型卡）+ 1 個本機實測代理。
實測代理寫了 `docs/exploration/bench_sd.py`（每個配置開獨立行程量 VRAM 才乾淨），
結果在 `results.jsonl`：

| 配置 | 步數 | 解析度 | ms/幀 | FPS | VRAM |
|---|---|---|---|---|---|
| LCM-LoRA | 4 | 512² | 226.2 | 4.42 | 3.4GB |
| LCM-LoRA | 2 | 512² | 149.9 | 6.67 | 3.4GB |
| **Hyper-SD 1step** | **1** | **512²** | **111.4** | **8.97** | **3.4GB** |
| Hyper-SD 1step | 1 | 512×768 | 164.4 | 6.08 | 3.7GB |

**SD1.5 + ControlNet + Hyper-SD 1 步可達 8.97 FPS**——技術上「準即時」成立，
但品質檔位（1 步的細節）與「8.97 FPS 算不算即時」的觀感風險都擺在桌上。

### 4.2 使用者轉向：「我想要更實用的東西」→ 2 代理證據淘汰

使用者看完實測後改變優先級：與其炫技，不如做畢業後拿得出手的實用工具。
主代理改開 **2 研究代理**：一個掃市場趨勢（電商、個人品牌的真實痛點），
一個做技術錨點查證（每個候選的關鍵權重、授權逐一驗證）。淘汰過程全憑證據：

| 候選 | 淘汰理由（查證代理的證據） |
|---|---|
| AI 證件照 | InsightFace 模型**非商用授權**，且人臉相似度不達標會被一眼看穿，demo 風險高 |
| 虛擬試穿 | 主流方案（如 IDM-VTON 級別品質的）**權重未完整公開**或授權受限，6/23 前不可控 |
| **商品攝影棚** | **入選**：rembg（MIT）+ IC-Light（碼 Apache-2.0）+ SD1.5/SDXL 全部本機拿得到；LLM 與 Diffusion 有真正的參數級咬合點（場景企劃 JSON → prompt + 光向） |

**SnapStudio 定案**：一張手機隨手拍 → 去背 → AI 場景 → 重新打光 → 多平台文案。

---

## 5. 階段二：凍結架構（ARCHITECTURE.md，6/10 12:06）

方向定案後，主代理把所有已驗證事實沉澱成 `docs/ARCHITECTURE.md` 並**凍結**，
作為後續所有建造代理的介面契約：

- **模組契約**：`matting / relight / scene / llm / pipeline / app` 六模組的
  輸入輸出與狀態標記（哪些 POC 已驗證、哪些待實作）
- **LLM JSON Schema**：商品卡／場景方案／文案包三個 schema，
  其中 `scene_prompt` 直餵 SDXL、`light_direction`/`light_desc` 直餵 IC-Light——
  這是「LLM 與 Diffusion 參數級咬合」的具體落點
- **里程碑**：M1（POC）～ M7（交付物），截止 6/23
- **降級鏈**：fbc 不穩退 fc、VLM 掛了退手動輸入、LLM JSON 飄移退預設模板——
  系統永不因單點故障卡死

凍結契約的目的：之後 8 個建造代理平行開工時，彼此不需要溝通，
只要各自符合 ARCHITECTURE.md 就能拼起來。

---

## 6. 階段三：IC-Light 在 diffusers 0.39 的重新實作（核心瓶頸）

這是全專題最硬的一塊。IC-Light 官方程式碼釘死 `diffusers==0.27`，
本機是 `0.39.0.dev0` 且**不准降版**（會連動 torch/transformers）。
主代理指派一個建造代理讀官方 demo 原始碼，把機制拆成三件事移植
（完整改寫指令見第 8 節，成果在 `poc/poc_iclight.py`）：

1. **UNet `conv_in` 4→8 通道**（fbc 模式為 4→12）：
   新建 `Conv2d(8, ...)`，前 4 通道複製原權重、新通道補零，bias 照搬——
   讓前景條件 latent 能與噪聲 latent 在通道維串接。
2. **權重 offset 合併**：IC-Light 釋出的 safetensors 不是完整權重而是 **offset**，
   必須逐 key `原版 + offset` 後 `load_state_dict(strict=True)`。
3. **forward 攔截**：0.39 的 `StableDiffusionPipeline` 不認識額外條件 latent，
   解法是包一層 `hooked_forward`，從 `cross_attention_kwargs` 偷渡 `concat_conds`，
   每步在 channel 維串接後呼叫原 forward——不用 fork 整條 pipeline。

途中踩到一個典型陷阱：**新建的 `conv_in` 預設 fp32，與 fp16 管線相撞**，
報 `RuntimeError: Input type (c10::Half) and bias type (float)`。
代理定位後一行修復：`unet.conv_in = new_conv_in.to(unet.dtype)`，
並把這條寫進 ARCHITECTURE.md 的「已驗證技術事實」防止重踩。

**實測結果（RTX 3090）**：768×768、25 步、**3.1 秒/張、峰值 VRAM 3.6GB**；
皮夾的皮革紋理與縫線完整保留，光向與色溫確實重算
（`examples/out_warm_window.png`、`examples/out_studio_rim.png`）。
M1 里程碑當日達成。

---

## 7. 專節：Agent 協助解決的技術瓶頸

| # | 瓶頸 | 診斷方式（代理做了什麼） | 解法 | 教訓 |
|---|---|---|---|---|
| 1 | HF CDN 長連線停滯，`from_pretrained` 線上下載卡死 | 實測代理每隔 8 秒量下載中檔案的大小，**增量為 0**，確認是連線停滯不是慢 | `wget -c --tries=50 --read-timeout=20 <resolve URL>` 斷點續傳（實測 **86.7MB/s**，對比原本卡死）；執行全程 `HF_HUB_OFFLINE=1`（已固化進 `snapstudio/config.py`） | 「下載很慢」要先量化成數字才能對症下藥 |
| 2 | 研究代理稱 controlnet_aux DWPose「走 onnx 可直接用」 | 查證代理直接讀 `/workspace/.venv-1` 內的原始碼，發現 import 鏈拉 mmcv/mmdet/mmpose 全家桶 | 判定為**研究幻覺**，改 YOLO11-pose + 手寫 COCO17→OpenPose18 對映（`poc_part4.py`） | 網路研究結論一律要過「本機原始碼查證」這關 |
| 3 | ControlNet `conditioning_scale=0.8` 姿勢被場景 prompt 拉走 | 實測代理固定 seed 做 0.8 vs 1.0 對照（`poc_part3.py`、兩張 grid） | 調到 **1.0** 完全鎖住姿勢 | 控制強度參數要用固定 seed 的 A/B 圖說話 |
| 4 | SDXL base + ControlNet 雙 pipeline 同駐 → CUDA OOM | 讀 OOM traceback + `max_memory_allocated` 量測 | 單管線配置（`poc_part2.py`），峰值 **12.3GB** | POC 腳本的記憶體配置要貼近正式版 |
| 5 | IC-Light 官方碼釘 diffusers 0.27，本機 0.39 不准降版 | 建造代理通讀官方 demo 原始碼，拆出三個移植點（第 6 節） | conv_in 擴通道 + offset 合併 + forward 攔截；768² 25 步 3.1s、3.6GB | 「版本不相容」未必要降版，讀懂機制就能平移 |
| 6 | fp16 管線 × fp32 新建 conv_in 型別衝突 | 讀 `RuntimeError: Input type (c10::Half) and bias type (float)` 定位到新建層 | `new_conv_in.to(unet.dtype)` 一行修復，寫進架構文件防重踩 | 手動改網路結構時，dtype/device 要顯式對齊 |

另外兩個「非技術但同樣關鍵」的瓶頸由協作機制本身解掉：

- **方向選錯的風險** → 「半天 POC 定生死」+ 兩輪證據淘汰（第 3、4 節），
  讓專題在投入大量實作前就轉到了權重、授權、時程三者皆可控的題目。
- **多代理平行的介面漂移風險** → 先凍結 ARCHITECTURE.md 再開工（第 5 節）。

---

## 8. 關鍵 Prompt 範例

以下為實際下達給子代理的指令重現（語意忠實，字句經整理摘錄）。

### 8.1 查證代理：攔下 DWPose 研究幻覺的指令

```text
你是技術查證代理。前一輪研究報告宣稱「controlnet_aux 的 DWPose 走 onnxruntime，
不需額外依賴」。不要相信報告，直接讀本機已安裝的原始碼查證：
1. 找出 /workspace/.venv-1 中 controlnet_aux 的 dwpose 模組，列出它實際 import 什麼
2. 確認每個 import 在本環境是否可解析（嚴禁 pip install 任何會動到 torch 的套件）
3. 結論只准二選一：「可直接用」或「不可用，因為 X」，附原始碼行號為證
```

→ 結論是「不可用」：DWPose 的偵測器初始化依賴 mmcv 系列，幻覺被攔下。

### 8.2 評審代理：三視角 lens 設計（以教授視角為例）

```text
你是三位獨立評審之一，lens =「教授視角」：只評估提案是否完整覆蓋課程要求
（LLM prompt engineering 與 API 整合、Diffusion 客製 pipeline、推論加速、互動 UI），
以及技術深度是否撐得起期末專題。不要考慮工程時程與 demo 效果——那是另外
兩位評審的 lens，你們不會看到彼此的評語。對兩份提案各給 1-10 分加三行理由，
禁止平手，必須指出你認為的最大扣分點。
```

→ 三位評審各自獨立輸出，主代理只彙整分數（2:1），不覆寫任何評語。

### 8.3 建造代理：IC-Light 移植指令

```text
官方 IC-Light 釘死 diffusers==0.27，本機是 0.39.0.dev0，不准降版。
請讀官方 gradio_demo.py 的實作，把三件事移植到 0.39 的 StableDiffusionPipeline：
(1) UNet conv_in 從 4 通道擴成 8 通道：原 4 通道權重複製、新通道補零、bias 照搬
(2) IC-Light 的 safetensors 是 offset 不是完整權重：逐 key 與原版 state_dict
    相加後 load_state_dict(strict=True)
(3) 0.39 不會幫你傳條件 latent：攔截 unet.forward，從 cross_attention_kwargs
    偷渡 concat_conds，在 channel 維串接 sample 後呼叫原 forward
完成後用 examples/test_product_input.png 實測兩種光線 prompt，
回報每張秒數與峰值 VRAM。注意 HF_HUB_OFFLINE=1，權重已在 weights/ 下。
```

→ 產出 `poc/poc_iclight.py`，含 fp16 陷阱修復；3.1s/張、3.6GB 實測達標。

---

## 9. 階段三～四：建造工作流與本文件的後設紀錄

架構凍結、核心瓶頸打通後，主代理開出本專題最大的一次平行工作流：

1. **8 個建造代理平行**：各核心模組（matting / relight / scene / llm / pipeline）、
   Gradio UI、文件（README 與本文件）各由一個代理負責，
   彼此不通訊，只認 ARCHITECTURE.md 的契約與 `snapstudio/config.py` 的路徑。
2. **整合代理**：收齊 8 份產出後做端到端串接——一張照片進、素材包出。
3. **驗收代理**：UI 煙霧測試 + 文件與程式碼的一致性檢查。
4. **交付物審查**：對照課程要求逐項核對三份交付物。

**後設說明**：本文件本身就是這個工作流的產物——撰寫它的建造代理
（workflowlog agent）讀取 `docs/exploration/` 的佐證檔案與時間戳重建時間軸，
所以上述流程描述同時也是這份文件誕生過程的第一手紀錄。

### 一日時間軸（6/10，依佐證檔案時間戳）

| 時間 | 事件 | 佐證 |
|---|---|---|
| 上午 | 4 方向發想 → 使用者選 2 → 9 代理評選（2:1） | — |
| 10:00 | PosePainter POC 第一版（雙管線 OOM） | `poc_posepainter.py` |
| 10:05–10:10 | 單管線補跑、scale 0.8→1.0、換 YOLO11-pose | `poc_part2/3/4.py` |
| 10:22 | 端到端 v2 對照圖完成，POC 判「活」 | `07_grid_v2.png` |
| 10:44–10:46 | 即時化基準：Hyper-SD 1 步 8.97 FPS | `bench_sd.py`、`results.jsonl` |
| ~11:00 | 使用者轉向「實用」→ 2 代理證據淘汰 → SnapStudio 定案 | — |
| 11:54 | 專案目錄建立、權重以 wget 預下載 | `weights/` |
| 12:06 | ARCHITECTURE.md 凍結 | `docs/ARCHITECTURE.md` |
| 12:27 | `config.py` 落地（離線鐵則固化） | `snapstudio/config.py` |
| 下午 | IC-Light 0.39 移植跑通（3.1s/張）→ 8 建造代理平行開工 | `poc/poc_iclight.py`、`examples/out_*.png` |

### 心得：這套協作模式真正值錢的三件事

1. **便宜的否定**：研究代理 + 查證代理 + 半天 POC，讓「換題目」的成本
   從一週降到半天。本專題實際換了兩次方向，每次都有數據與授權證據背書。
2. **幻覺有專屬防線**：研究代理會自信地給出錯誤結論（DWPose 事件），
   但「讀本機原始碼查證」是結構性的攔截，不靠運氣。
3. **契約先行讓平行成為可能**：凍結 ARCHITECTURE.md 之後，8 個代理
   同時開工而不互相踩腳——人類團隊要開三次會才能做到的事，
   一份寫清楚的介面文件就解決了。

---

## 10. 迭代優化期（6/11–6/16）：多代理迴圈與 Agent 解瓶頸

第 9 節把 v2 雛形拼起來後，進入「使用者實測 → Agent 迭代」的長尾優化。這一期的協作工具
從「一次性平行建造」升級成 **可程式化的多代理工作流（Workflow）**：用 JS 腳本決定性地
編排 fan-out（多視角評審）、pipeline（逐品流水）、adversarial verify（找到的問題再派獨立
agent 反駁），並大量使用 **playwright 實際開瀏覽器看成品 / 截圖驗證**、**ffmpeg 做 demo**。

### 10.1 多代理品質迴圈：找圖 → 生成 → 三視角評審 → 修正

使用者要「幾乎所有展示品都達電商客戶滿意」。主代理開出可重跑的工作流（`examples/review/`）：
(1) 數個 source-finder agent 上網（Unsplash）找乾淨高解析來源圖、去背；(2) GPU 序列生成；
(3) **三視角評審 fan-out**——完美主義（品牌總監）／務實客戶／破綻獵人各一 agent 平行打分；
(4) 綜合 agent 把破綻對應到具體 `file::function` 修法。**工具組合**：Workflow（parallel 評審 +
pipeline 逐品）+ Bash GPU 生成 + 視覺 agent 讀圖。產出鐵律「**接地陰影必須程式合成、不能靠
prompt**」→ 固化進 `compose.paste_back` 三層陰影。

### 10.2 雙模式自動路由與重塑（穿戴/手持）

手錶鍊條摺疊去背醜 → 使用者洞見「讓生圖重塑產品戴上手腕」。實作 **IP-Adapter（SDXL
plus-vit-h）** 保留產品身份重畫姿態（`reshape.py`），並讓 **VLM 看圖自判 `product_class`**
驅動「剛性→鎖定／穿戴手持→重塑」自動路由（關鍵字僅離線備援）。對抗審查 workflow 抓出 6 個
真 bug（VLM 回表外類別被壓 rigid、refine 對 reshape 誤走 inpaint…）全修。

### 10.3 ★最硬的除錯：「不管下什麼 prompt 都極簡」三層根因

使用者回報：下「陽光花園」背景永遠是極簡灰棚，prompt 像被無視。這是全專題最深的一次
debug，靠 **Agent 化的科學方法**逐層剝開（不靠猜）：

- **隔離法（A/B/C）**：先用 agent 直接呼叫 `SceneInpainter` 證明「inpainter 生花園毫無問題」
  → 鎖定真凶在 full pipeline，而非 prompt 或模型。
- **攔截可觀測性**：包一層 `_chat_once` 印出每次 LLM 呼叫的 model 與真實例外 → 抓到鐵證
  `CHAT FAIL qwen3 500 model failed to load` 連續 6 次。
- **三層根因**：(1) scene_rule 把 `pure white seamless` 無條件加到每個 prompt 洗白；
  (2) ★**最底層**：`identify_product` 載入的 VLM（~21GB）用完沒卸，接著文字模型載入時與 VLM
  同駐爆 24GB → Ollama 回 500 → `_structured_call` 靜默退 `DEFAULT_SCENE_PLANS`（寫死的極簡
  模板）；(3) QC `judge` 一判 needs_fix 就把整個使用者場景換成無菌棚景。
- **修**：pipeline 在 identify 後、plan 前先 `release_models(wait=True)` 卸 VLM 騰 VRAM；
  scene_rule 改有條件 + 使用者需求最優先；QC 改為保留原場景只加隔離負面詞。每步都派**獨立
  視覺 agent 核實**成品真的出現花園 + 產品為主角。

### 10.4 文字模型換 qwen3:32b + 規則瘦身（少硬規則）

使用者多次要求「多用 LLM、少寫硬規則」。查證發現遠端 Big Pickle（opencode zen → GLM/
deepseek 系推理模型）**時通時斷、常回空 content**，長期其實在用較弱的 14b 備援 → 規則越堆
越多、甚至用新規則抵消舊規則（soft diffused ↔ visible sunbeam），這是「過度限制」的徵兆。
**決策**：文字主力改本機 **qwen3:32b**、遠端預設關（`LLM_USE_REMOTE`）、scene_rule 從 62 行
碎念瘦身成 7 條原則，把判斷交給更強的模型。

### 10.5 前端 premium 改版（研究 workflow + playwright 驗證）

使用者要「設計公司美感」。主代理開 **研究 workflow**：一 agent 上網研究 Linear/Vercel/Stripe
的設計 tokens、一 agent 查證 Gradio 6.17 主題/CSS 可客製範圍、一 agent 盤點 app.py 結構 →
綜合成可落地規格。實作自訂 `gr.themes.Base`（暗色攝影棚 + 暖琥珀）。**踩坑**：字體混用
`GoogleFont`+字串會觸發 gradio 主題比較的 `Font.__eq__` 對字串取 `.name` 崩潰 → 改用
`launch(head=)` 注入 `<link>` + CSS 變數 `--font`。**用 playwright headless 截圖目視驗證**
（`networkidle` 不適用 gradio 的常駐 websocket → 改 `domcontentloaded`）。

### 10.6 IC-Light A 護字（feasibility workflow → 純 CV 實作）

使用者要「IC-Light 但保住標籤文字」。主代理開 **可行性 workflow**：查本機文字偵測能力、
relight 合成注入點、自動 gate 設計。**關鍵澄清**（讀碼後修正前提）：鎖定模式 `paste_back`
本就把整個產品原像素貼回、IC-Light 只重打背景 → 文字本來就沒被糊。真正價值是讓產品**表面**
吃場景光。實作 `compose.harmonize_keep_text`：純 CV（借 `reshape._dominant_hue/_face_region`，
零模型下載）偵測彩色標籤面 → 在「內縮輪廓 ∖ 文字區」把成品往 relit 版混 alpha，**三層
fallback**（偵測不可靠/例外 → 退回原始貼回，永不更差）。

### 10.7 這一期新解掉的技術瓶頸

| # | 瓶頸 | 診斷方式（Agent 做了什麼） | 解法 |
|---|---|---|---|
| 7 | 下任何 prompt 背景都極簡 | A/B/C 隔離 + 攔 `_chat_once` 印例外 → 抓到 500 | identify 後先卸 VLM 騰 VRAM；prompt 有條件化；QC 不洗場景 |
| 8 | Ollama VLM+文字模型同駐爆 24GB | 對照「隔離跑正常 / full pipeline 退 default」+ log 鐵證 | `release_models(wait=True)` 序列化單卡模型 |
| 9 | 規則越堆越多、自相矛盾 | 發現新規則在抵消舊規則 | 換更強模型（32b）+ 砍規則，而非繼續加 |
| 10 | Gradio 主題字體設定崩潰 | 讀 traceback 定位 `Font.__eq__` | 字體改 `launch(head=)` 注入 + CSS `--font` |
| 11 | 多個 app 殘留互搶 GPU、生成卡 700s | `nvidia-smi`/`ss` 查兩個 app + 兩 ollama runner 佔滿 | 殺重複 app + 卸 ollama；按 port 殺避免 `pkill -f` 自我匹配 |

### 10.8 這一期的關鍵 prompt 範例

```text
（隔離除錯指令）逐層找出「下任何 prompt 背景都極簡」的真因，不要猜。
先直接呼叫 SceneInpainter 證明 inpainter 本身能不能生花園（排除模型/prompt），
再攔截 _chat_once 印出每次 LLM 呼叫的 model 與真實例外，把 full pipeline 與隔離跑的差異列出來。
```
```text
（前端研究 workflow）平行三 agent：一個上網研究 Linear/Vercel/Stripe 等高質感站的設計
tokens（配色 hex/字體/圓角陰影/留白），一個查 Gradio 6.17 的 theme/CSS 實際可客製到哪、
哪些做不到（別過度承諾），一個盤點現有 app.py 結構與注入點，最後綜合成可落地的設計規格。
```

### 心得補充（迭代期）

- **可觀測性優先於猜測**：最深的 bug（VLM 500 退 default）肉眼完全看不出來，是「攔 `_chat_once`
  印例外」這個 5 行的觀測手段揪出來的。
- **規則打架 = 該換模型而非加規則**：用新規則抵消舊規則時，根因通常是底層能力不足。
- **每個成品都派獨立視覺 agent 核實**，不靠主代理自說自話——這是這一期的品質鐵則。
