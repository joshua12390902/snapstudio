# SnapStudio 開發日誌（問題 → 嘗試 → 解決）

逐條記錄遇到的問題、當下的假設、怎麼驗證、最後怎麼解。重點是**思考過程**，
包含走錯的路與被推翻的假設——那些往往比結論更有參考價值。

---

## 1. 手錶「巨錶貼在臉上」

- **現象**：戴上身模式生出「巨大手錶蓋在一張人臉上」。
- **假設順序**：
  1. 以為是遮罩抓錯位置 → 看 `wornplace.body_part_mask`，它取「最大膚色塊」。
  2. 但更上游：先看 AnyDoor 的「裸場景」是什麼 → **直接 Read 場景圖**，發現生出來是
     **人臉肖像**，根本沒有手腕。
- **根因**：VLM 把平放桌上的折疊手錶判成 `rigid`、`worn_framing` 留空 → fallback 只修了
  類別沒修取景 → `framing_for` 退回「a person」→ text2img 生人臉 → 遮罩只能落在臉上。
- **解**：(a) VLM prompt 要求「依產品本質而非當前擺放判斷、wearable 必填 worn_framing」；
  (b) `framing_for` 備援改「手腕特寫」絕不用「a person」；(c) `bare_scene_prompt` 以身體
  部位為主體、負面詞擋人臉。
- **教訓**：不要從最靠近現象的那一層開始猜，先沿資料流往上游看「輸入長怎樣」。

## 2. 「雙錶／巨錶」

- **現象**：AnyDoor 把錶擺上手腕，但常複製出第二支、或尺寸過大。
- **驗證**：量遮罩 bbox → 發現某張遮罩是 **679×766（蓋掉半張圖）**；好的約 200px。
- **根因有兩個**：
  1. 遮罩尺寸 `rad = arm_w × scale`，`arm_w`（肢寬）在粗手臂近拍會量到大半張圖 →
     遮罩失控。**改畫面比例化** `rad = 0.22×max(H,W)×scale`（場景已固定「部位填滿畫面」，
     比例化才泛化）。
  2. 即使遮罩正常，AnyDoor 對「折疊手錶」參考圖（錶頭＋拖在後面的折疊錶帶）會把錶帶
     扣環誤生成第二個圓盤。**做 `compact_reference` 裁到產品主面**。
- **被推翻的假設**：以為「縮小遮罩」就能解雙錶 → 實測 shot_0 在 224px 也雙錶 → 不是純尺寸，
  是參考圖形狀。
- **延伸**：`compact_reference` 用色彩偵測錶盤來裁，但這支**暗藍盤飽和度不足、偵測不到** →
  退回原圖。最後是「畫面比例化的小遮罩」讓 AnyDoor 不再平鋪複製，解掉雙錶。

## 3. 手持（controller）「兩隻手空抓」很恐怖

- **現象**：手持模式生出兩隻手往上抓空氣，詭異。
- **根因**：裸場景把「two hands holding the controller」去掉動詞後變「two hands」→ 生出
  「握著空氣」的手；空握比裸手腕難看得多。
- **解**：`bare_scene_prompt` 對手持類改「攤開放鬆、掌心朝上呈現」→ AnyDoor 把產品擺在攤開
  的掌上才自然。穿戴部位（耳/腕/指）優先保留，不被 hand 規則蓋掉。

## 4. 第二次生成 SAM2 去背 OOM

- **現象**：連續生成第二次時，最前面的 SAM2 去背 CUDA OOM、退回較糙的 rembg。
- **驗證**：看 log「Process xxxx has 22.22 GiB」→ 是 Ollama 的 VLM 沒釋放。
- **根因**：重塑結尾 `judge_worn` 會重載 VLM，但那條路跑完**沒再 release** → VLM 賴在卡上。
- **解**：`process()` 首尾都 `release_models()`；並給它加 `wait`（輪詢 /api/ps 直到真的卸完）。

## 5. VLM 挑最佳參考角度（pick_reference）

- **動機**：使用者會上傳多角度照。折疊角度當參考會害 AnyDoor 失真，正面照穩很多。
- **解**：在 identify 階段（VLM 仍載入）讓 VLM 從多張去背圖挑「最正面、最完整」那張當
  AnyDoor 參考。實測：故意丟折疊照當主圖，VLM 仍正確挑出正面照。

## 6. 戴上身「浮空、沒真的戴上腕」

- **現象**：某些 seed 的手勢沒把手腕擺出來，錶就被擺到浮空。
- **解**：`judge_worn` 重試擴充——除「太大→縮小遮罩同場景重擺」，新增「placement off 或
  not natural →換 seed 重生場景再擺」（縮放治不了壞手勢，得換場景）。
- **代價**：harsh VLM 會把 3 張都判 off → 全部重生 → 較慢但品質穩。

## 7. 嚴格 VLM 審查員「校準過嚴」

- **發現**：拿 qwen2.5vl:32b 當「極嚴格廣告總監」評分，**連我們驗證為乾淨的手錶也打 3 分**
  （理由是「材質紋理不夠真實」這種 AI 痕跡），但香水卻給 9 分。
- **結論**：它**不適合當絕對 pass/fail 門檻**，但拿來抓「具體破綻」很有用（靠它抓到 controller
  空抓、變異）。→ 用它做「相對比較＋找破綻」，不是「絕對及格線」。

## 8. 手錶戴上身的寫實天花板 → 改走鎖定擺台

- **現象**：結構問題（貼臉/雙錶）都修好後，使用者仍覺得「錶太大、金屬錶帶假/糊」。
- **判斷**：金屬錶帶反光複雜，AnyDoor **重繪**就是會假——這是路線天花板，用擺位/尺寸救不了。
- **對照**：生鎖定模式手錶英雄照 → 錶是**真實像素**（錶帶/錶盤/logo 全真），只有背景生成 →
  乾淨得多。**決策：手錶改走鎖定模式。**

## 9. 鎖定模式超慢（2 張 26 分鐘）

- **第一假設**：以為是 `_render_plan` 每張在 inpaint↔IC-Light 之間換模型（O(n) 次換）。
- **驗證**：讀 `_ensure_relight` → **它根本不卸 inpaint**（line 88-93 只在 None 時建 Relighter，
  沒有 unload inpaint），所以 inpaint 與 relight **共駐**，沒有 per-shot thrash。**假設被推翻。**
- **真因（待測量確認）**：VLM 32b 載入兩次（identify + QC judge）＋ QC 對 harsh 判定狂重生。
  鎖定品本來就乾淨（像素鎖死），QC 高成本低價值。
- **計畫**：先把 QC 改保守（只在明顯破綻 needs_fix 且分數低才重生），再加階段計時實測真因。

## 10. 鎖定批次 watch 又 OOM（release race condition）

- **現象**：全展示品批次第一個 watch(鎖定) 就 CUDA OOM，「Process xxx has 22.22 GiB」＝VLM 沒卸。
- **困惑**：我明明在 process() 開頭加了 `release_models(wait=True)`，怎麼還 OOM？
- **追查**：`grep release_models pipeline.py` 列出所有呼叫點 → 發現**關鍵那個不是開頭那個**。
  開頭那個在 identify 前(此時還沒載 VLM，no-op)；真正要等的是 **identify 之後、載 diffusion 前**
  的 line 288，那行是舊的 `release_models()` **沒 wait** → Ollama keep_alive=0 只是「排程卸載」、
  馬上 `_ensure_inpaint()` 載 diffusion → Ollama 還沒卸完 → OOM。
- **為何時好時壞**：race。之前單張 e2e、6 品批次剛好卸得夠快就過；這次沒過。**race 要根除不能靠運氣。**
- **解**：所有「載 diffusion/AnyDoor 前」的 release 都改 `wait=True`(line 288/355/432)；
  且 `release_models` 偵測 /api/ps 清空後再 `sleep(1.5)` 讓 CUDA 記憶體實際釋放(回空到實際釋放有延遲)。
- **教訓**：(1) 同一個 bug 修了「一個」呼叫點不代表修完，要 grep 全部呼叫點看哪個才是關鍵路徑；
  (2) 「時好時壞」≈ race，別當成已修好，要找到決定性的同步點。

## 11. 鎖定模式 26 分的「真‑真因」：32b VLM 判圖單次 294 秒

- **又一次假設被推翻**：先以為慢是 per-shot 換模型(§9 推翻)，再以為是 QC 狂重生。**直接寫
  獨立計時測試**才水落石出：單一 `judge_product_shot`(32b VLM, 1024 高的圖) 竟花 **294 秒**。
- **真因**：32b 視覺模型對大圖推論本來就很慢。鎖定 2 張 = identify + 2 judges ≈ 15 分 VLM。
- **解**：
  1. 裁判改用 **qwen2.5vl:7b**(`OLLAMA_JUDGE_MODEL`)，識別/挑參考仍用 32b 求準。
     實測判圖 **294s→62s 冷 / 0s 熱**，且 7b 沒誤判乾淨香水。
  2. 判圖縮到 `max_side=672`(裁判只需抓明顯破綻，不需 1024)。
- **順帶根治 VRAM OOM**：`release_models(wait)` 不再信 Ollama /api/ps「回報清空」(假訊號，
  VRAM 還沒吐)，改**輪詢 nvidia-smi 實際剩餘記憶體 >16GB 才返回**。實測 keep_alive=0 後
  1 秒就把 22GB 吐乾淨。
- **教訓**：效能問題**先量測再優化**。我前兩個假設(換模型、狂重生)都錯，寫一個 30 行的獨立
  計時腳本就直接指出真兇，省下無數瞎猜。

## 12. 「修了卻一直 OOM」的幽靈：zsh noclobber 害我讀到舊 log

- **現象**：明明把 release 改 wait、改 nvidia-smi、判圖改 7b，重跑批次卻**每次都同一個 OOM，
  連 PID(4025230/4028183) 和位元組數都一字不差**。三次獨立執行不可能位元組全同。
- **追查**：跑單一測試時 shell 報 `(eval): file exists: /tmp/gpu_trace.log`。**恍然大悟**：
  這台的 shell 是 **zsh 且開了 `noclobber`，`>` 不能覆蓋已存在的檔案**。我每次「重啟批次」用
  `> /tmp/batch_v2.log` 都因 noclobber **失敗、python 根本沒重跑**，於是一直讀到**第一次**
  失敗時寫下的舊 log → 看到一模一樣的 OOM。**我的修正從來沒在批次裡真正跑過，全程在追幽靈。**
- **解**：覆蓋檔案前先 `rm -f`（或用 zsh 的 `>!`/`>|`）。
- **教訓**（最痛）：(1) 「結果一字不差地重現」不是「bug 還在」，要先懷疑**根本沒重跑**；
  (2) 環境的 shell 行為(noclobber)會靜默吃掉你的指令；(3) 看到無法解釋的「完全相同」就去質疑
  測試管線本身，而不是反覆改被測程式。

## 13. identify 也要 900s？→ 32b CPU offload 的決定性證據與解法

- **現象**：批次裡 `identify` 計時 **900s**(≈3×300s client timeout)，watch/perfume 都一樣。
- **量測**：寫診斷腳本同時測 32b vs 7b 並印 `ollama ps` 的 vram/size：
  - 32b：identify **263s**，`size=51.1G vram=24.4G` → **CPU offload**(只 24GB 進卡，其餘 CPU)。
  - 7b：identify **7s**，`size=22.8G vram=22.8G` → **全進 GPU**，且分類同樣正確(wearable/PRX)。
- **關鍵領悟**：32b 慢不是模型「重」，是**裝不下被迫 offload**。51GB 大半是 context KV cache。
  測縮 `num_ctx`：4096→9s、8192→23s（offload 大幅縮小）。
- **解（兼顧使用者要的「強 VLM」）**：建 `qwen2.5vl:32b-ctx8k`(num_ctx 8192)。實測 identify 29s、
  輸出比 7b 強(會填對 worn_framing)。`_ensure_vision` 偵測 `-ctx` 變體會自動 `ollama create`。
  要更快可改 7b。
- **連帶**：把單品總時間從 581~1391s 砍到 ~206s(7b)/~330s(32b-ctx8k)；identify 263→29s 是主刀。
- **教訓**：VLM「越大越好」在固定 VRAM 下是假的——**裝不下就 offload，大模型反而慢 10 倍**。
  要嘛挑裝得下的尺寸，要嘛縮 context 讓大模型擠進 GPU。先看 `ollama ps` 的 vram/size 比才知真相。

## 14. 16-agent Workflow 嚴格審查 + 鎖定模式「產品被延伸/外溢」根因

- **做法**：全 8 品生成後，跑 Workflow fan-out 16 個 Claude 視覺 agent(每品「完美主義者+務實
  電商客戶」雙視角)+1 綜合。結果：5 接受(watch/perfume/energy/lipstick/controller)、3 拒
  (sneaker/wallet/earbuds)、整理出 6 大系統性瑕疵。
- **最反直覺的指控**：reviewer 說「鎖定模式產品本體被重畫」——錶變 TV 方盒、香水長蠟燭、
  皮夾蛇皮淹滿整個廚房中島。**我親自開圖驗證→屬實，reviewer 沒誇大。**
- **dump 中間圖找機制**：dump cutout/mask/product/inpaint原始/paste_back後。發現：
  **paste_back 沒失敗**，產品本體有貼回；是 **inpaint 在「產品緊鄰的場景區(白遮罩)」自動
  延伸/補全產品**——錶旁邊長出錶冠球、香水瓶口長蠟燭、皮夾把蛇皮紋外溢鋪滿場景。遮罩鎖住
  產品「像素」，但管不住 inpaint 在旁邊「接著畫」。
- **解(已實測)**：加 `LOCKED_NEGATIVE` 通用負面詞(extra parts/crown/candle/repeated texture/
  product texture on walls…)。手錶實測 2 seed→錶冠球消失、乾淨完整。
- **教訓**：「鎖定/凍結」只保證被遮罩的像素不變，**不保證旁邊不會生成與產品相關的東西**；
  延伸/外溢要靠負面詞與場景約束擋。驗 reviewer 指控一定要自己開圖——但這次它對了。

---
（持續更新）

## 15. 兩輪 Workflow 重審迭代：鎖定模式從「會毀產品」到「6/8 紮實」

- **R1→R2**：加 LOCKED_NEGATIVE(通用物理) + plan_scenes 禁道具/禁改造產品風格 + judge 強化抓
  morph + QC 改信 needs_fix 並「重生用乾淨棚景」。R2 重審：5/8 接受，watch morph、wallet 蛇皮
  淹滿都修掉(但 wallet 變過大方箱)。
- **R2→R3**：(1) compose.py 遮罩「先 MaxFilter dilate 再羽化」——羽化會往產品內側吃、去背對細長
  錶帶易漏，導致 inpaint 碰產品邊緣染色(watch 橘藍塊/wallet 鏽邊/lipstick 接縫)。dilate 補償後
  watch_p1 變乾淨且 TISSOT logo 清晰可讀。(2) plan_scenes product_scale 依產品真實大小給(小件
  0.30~0.40)，解 wallet 過大方箱→變乾淨小皮夾。
- **現況**：watch/wallet/perfume/lipstick/energy_drink/controller 6 品紮實可用；sneaker/earbuds
  是 reshape 對「腳/耳」的模型天花板(多腿/壞手/浮空/貼臉頰)，非 prompt 可救，待產品決策(導向乾淨
  locked 擺台 vs 續推 worn)。
- **方法論收穫**：多 agent 雙視角審查(務實客戶當及格線+完美主義者抓破綻)能穩定指出系統性瑕疵；
  每輪「審查→改 code→重生→再審」確實收斂。VLM 驅動(judge 抓破綻→重生)+ 最小通用負面詞，比硬列
  產品專屬規則更符合需求且有效。

## 16. Round 4-5：IC-Light off 保 logo + 修自引入的彩色邊暈

- **亂碼字根因(dump 證實)**：energy_drink 去背圖 logo 清晰可讀，但成品變 MONƎTER——**不是來源糊，
  是 pipeline 弄壞**。隔離測試：harmonize=False(關 IC-Light)→logo 恢復可讀。**IC-Light 光線融合
  會洗糊/扭曲產品正面文字**。→ 鎖定模式預設關 IC-Light(接地靠程式陰影、不靠它)。
- **VLM best_shot 路由**：加 ProductCard.best_shot，VLM 自判 clean/worn(穿戴不可靠的鞋/耳機→clean)，
  sneaker/earbuds/controller 全自動走乾淨 locked → 不再浮空/貼臉。少硬規則、交給 VLM。
- **自引入回歸**：§15 的遮罩 dilate(MaxFilter)雖修了邊緣染色，卻讓「擴張鎖定環」殘留高頻雜訊→
  成品產品輪廓一圈彩色 confetti(IC-Light 開時被糊掉、關後曝露)。**修：擴張環在 init 先鋪 127 灰**，
  產品蓋回後只剩中性灰環被鎖、羽化後柔順銜接→彩邊消失。
- **5 輪迭代收斂**：1→5 輪「生成→16-agent 雙視角審查→改 code→重生」，務實客戶可用度從 5/8 到
  穩定 6/8(watch/perfume/energy/sneaker/wallet/earbuds)。剩 lipstick(透明殼歧義)、controller
  (來源圖本身是怪工業盒非正常手把，garbage-in)。
- **誠實天花板**：100% 每張完美受限於 (a)inpaint 偶發在產品旁長附加物(seed 變異，靠多生挑最佳)、
  (b)極小字 logo 放大檢視的亂碼(來源解析度)、(c)個別來源圖品質(controller)。非單一 bug 可全解。

## 17. 多 agent 全迴圈：找新圖→生成→3視角裁判→修正→再生→再審

- **使用者要的完整機制全部跑通**：(1) 5 個找圖 agent 上網(Unsplash)找乾淨高解析來源(DualShock4/
  Sony耳機/墨鏡/真皮包/勞力士)，全成功下載+PIL+Read 驗證；(2) serial GPU 生成；(3) 3 視角多裁判
  (完美主義/務實客戶/破綻獵人)平行審查；(4) 綜合提具體檔案::函式修法；(5) 修 code→重生→再審。
- **本輪修正**：best_shot 強偏 clean(worn 全品類不可靠)、paste_back 去色邊(MinFilter5+邊緣去飽和)。
  sunglasses/watch2 改 clean → 產品回來、不再雙錶。
- **務實客戶標準下**：controller/headphones/watch2 通過(8-9 分)。
- **誠實天花板(剩餘長尾)**：
  - sunglasses 細鏡框：去背難精準切細結構→框型變形。
  - handbag：inpaint 在包體旁「補出」鬼影側袋(extra-parts 延伸，負面詞未完全壓住)。
  - 彩色邊暈：**次知覺等級**——破綻獵人用像素測色度(背景3.5 vs 接縫17)仍報，但務實/完美視角都判乾淨採用。追到此屬過擬合鑑識 agent，非使用者的電商標準。
  - 根因多指向 matting.py 對細長/複雜輪廓的覆蓋——屬核心改動、風險高、收益遞減。
- **可重用工具**：examples/review/{source_finder,review3,gen_sourced}——可隨時再跑更多迴圈。

## 18.「不管下什麼 prompt 都極簡」——三層根因 + 改用 32b + 規則瘦身

- **使用者回報**：下 brief（如『陽光 花園』『窗戶灑進陽光』）背景永遠是極簡灰棚，prompt 像被無視。
- **診斷方法（不靠猜）**：攔截 `_chat_once` 印出真實例外 + A/B/C 對照（先用直接呼叫 SceneInpainter
  證明「inpainter 生花園毫無問題」→ 鎖定真凶在 full pipeline，而非 prompt 或 inpaint 本身）。
- **三層根因（由淺到深）**：
  1. `plan_scenes` scene_rule 把 `pure white RGB 255 255 255 seamless` **無條件**加到每個
     scene_prompt 結尾 → 純白把豐富場景洗白。修：改成只有使用者明確要純白才寫，否則禁出現
     white/seamless/studio 字眼，並把「忠實反映使用者需求元素」設最高優先。
  2. **（最底層、最關鍵）VRAM 500 陷阱**：`identify_product` 載入 VLM `qwen2.5vl:32b`（~21GB）
     用完**沒卸**，接著 `plan_scenes`/`write_copy` 要 Ollama 載 qwen3 文字模型（~9-20GB）→ 同駐爆
     24GB → Ollama 回 `500 model failed to load` → `_structured_call` 靜默吞例外回 None →
     `plan_scenes` 退 `DEFAULT_SCENE_PLANS`（寫死的 minimalist seamless gray）。隔離跑沒 VLM 卡著
     就正常，極難察覺。修：pipeline 在 identify 後、plan_scenes 前先 `release_models(wait=True)` 卸
     VLM 騰 VRAM。log 鐵證：連 6 次 `CHAT FAIL 500`，且輸出逐字等於硬編碼模板。
  3. QC `judge_product_shot` 判 needs_fix 時，原本把**整個使用者場景**換成無菌棚景 prompt → 洗掉
     花園。修：保留原 scene_prompt，只加「道具別碰產品」隔離負面詞；judge 也澄清「豐富背景不是
     破綻，只有道具貼到/壓到產品才算」，避免誤判豐富背景而觸發重生。
- **接著做品質升級（每步都派獨立 Claude 視覺 agent 核實）**：
  - 產品主角化：product_scale 偏 hero、產品須為全圖最醒目、道具退陪襯（更小更虛、不喧賓奪主）。
  - 擺位合理：嚴禁產品擺地面/草地，必須站在正下方抬高檯面（agent 抓到「香水擺戶外地上」）。
  - 淺/逆光背景產品要跳出來：加 rim light 分離輪廓、正前方淨空（agent 抓到「淺瓶配淺磚牆糊在一起」）。
  - 使用者點名的光要看得見：brief 寫『窗戶灑進陽光』卻被先前「一律 soft diffused」(防浮空) 壓成平光
    → 加規則「點名光線現象時寫成 visible sunbeams/god rays，不降級」。實測 sun_1 出現明顯窗光。
- **Big Pickle 真相**：遠端文字主力 `opencode.ai/zen` 的 `big-pickle`（實際路由到 deepseek-v4-flash）
  並非掛了——網路通、會回應，但它是**推理模型常回空 content**、且時通時斷，`_ensure_pickle` 探測常
  誤判成不可用 → 長期其實在用備胎 14b（笨 → 規則越堆越多）。
- **收斂（呼應使用者「少寫硬規則」）**：(1) `OLLAMA_TEXT_MODEL` 14b→**qwen3:32b**（本機、更聽話）；
  (2) 新增 `LLM_USE_REMOTE` 開關**預設關**飄忽的遠端；(3) scene_rule 從 62 行碎念**瘦身成 7 條核心原則**
  （主角/接地單光源/忠實 brief/對比/背景呼應/變化/格式），把判斷交給更強的模型。實測 32b+瘦身規則
  對『窗戶灑進陽光』→ 兩方案皆桌面+窗光(PLAN1 direct sunbeam)+植物、產品為主角、無退步。
- **代價/取捨**：32b 較慢（2 方案全流程 ~467s，約 14b 兩倍），但更聽話、規則打架少。已做成環境變數
  `SNAPSTUDIO_OLLAMA_TEXT_MODEL` 可切；預設 32b（品質優先）。
- **教訓**：① 任何「輸出變成 DEFAULT 模板」的詭異現象，先查 Ollama 是否 VLM+文字模型同駐 500；
  ② 規則越堆越多、甚至用新規則抵消舊規則（soft diffused ↔ visible sunbeam），是「過度限制」的徵兆，
  解法是換更強模型 + 砍規則，而非繼續加。
