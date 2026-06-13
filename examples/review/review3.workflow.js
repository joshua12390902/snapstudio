export const meta = {
  name: 'snapstudio-review3',
  description: '3 視角多裁判嚴格審查 + 提出具體 code 修法的綜合',
  phases: [
    { title: 'Review', detail: '每品 3 視角(完美主義/務實客戶/破綻獵人)平行審查' },
    { title: 'Synthesize', detail: '綜合 + 對應到具體檔案/函式的修法' },
  ],
}

const PRODUCTS = (typeof args === 'string' ? JSON.parse(args) : args) || []

const SHOT = {
  type: 'object',
  required: ['client_accept', 'score', 'verdict', 'flaws', 'worst_flaw_type'],
  properties: {
    client_accept: { type: 'boolean' }, score: { type: 'integer' },
    verdict: { type: 'string' }, flaws: { type: 'array', items: { type: 'string' } },
    worst_flaw_type: { type: 'string', description: 'doubling|floating|seam|distortion|gibberish_text|texture_bleed|extra_object|grounding|lighting|reflection|none|other' },
  },
}
const REVIEW = {
  type: 'object', required: ['name', 'shots'],
  properties: { name: { type: 'string' }, shots: { type: 'array', items: SHOT },
    root_cause: { type: 'string' }, fix: { type: 'string' } },
}

const LENS = {
  perfectionist: '你是頂尖品牌(Apple/精品)的廣告攝影總監，標準是「首頁英雄大圖」。吹毛求疵。',
  client: '你是務實的電商賣家客戶：圖只要乾淨、專業、無明顯 AI 破綻就採用，不追求藝術完美。',
  defect: '你是 AI 影像鑑識專家，**專門放大找破綻**：產品輪廓彩色邊暈/接縫、亂碼假字、浮空無接地、重複/第二物件、產品旁長出附加物、材質滲入背景、倒影中文字扭曲、邊緣鋸齒。逐項掃描。',
}
function prompt(p, lens) {
  const imgs = p.paths.map((x, i) => `第${i + 1}張: ${x}`).join('\n')
  return `${LENS[lens]}
商品「${p.name_guess || p.name}」的 AI 生成電商成品。用 Read 逐一打開**親眼看**：
${imgs}
判斷標準(電商可用)：產品不變形/不被改造/比例對；無雙重或附加物件、無彩色邊暈/接縫、無亂碼假字、
無浮空、無材質滲背景、倒影正常。對每張回一個 shot(順序對應)，client_accept=務實客戶會否採用。繁體中文、不要簡體。`
}

const reviews = await pipeline(PRODUCTS, async (p) => {
  const [a, b, c] = await parallel([
    () => agent(prompt(p, 'perfectionist'), { label: `perf:${p.name}`, phase: 'Review', schema: REVIEW }),
    () => agent(prompt(p, 'client'), { label: `client:${p.name}`, phase: 'Review', schema: REVIEW }),
    () => agent(prompt(p, 'defect'), { label: `defect:${p.name}`, phase: 'Review', schema: REVIEW }),
  ])
  return { name: p.name, perfectionist: a, client: b, defect: c }
})

const SYNTH = {
  type: 'object', required: ['accepted', 'rejected', 'fixes'],
  properties: {
    accepted: { type: 'array', items: { type: 'string' }, description: '務實客戶可接受(≥1張)的產品' },
    rejected: { type: 'array', items: { type: 'string' } },
    fixes: { type: 'array', items: { type: 'object', required: ['issue', 'affected', 'where', 'how'],
      properties: { issue: { type: 'string' }, affected: { type: 'array', items: { type: 'string' } },
        where: { type: 'string', description: '具體檔案::函式(如 compose.py::build_scene_inputs)' },
        how: { type: 'string', description: '具體怎麼改' } } },
      description: '依影響力排序的可執行修法' },
  },
}
const synthesis = await agent(
  `以下是各商品的 3 視角審查(JSON)。及格線用務實客戶 client_accept(≥1 張可用即 accepted)，但把破綻獵人/完美
主義者抓到的可修破綻整理成**可執行修法**，每項對應到 SnapStudio 的具體檔案/函式(compose.py/groundgen.py/
llm.py::plan_scenes/judge_product_shot/pipeline.py/matting.py)並寫清楚怎麼改。資料：
${JSON.stringify(reviews, null, 2)}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH }
)
return { reviews, synthesis }
