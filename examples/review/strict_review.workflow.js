export const meta = {
  name: 'snapstudio-strict-review',
  description: '嚴格審查 SnapStudio 各展示品成品，雙視角校準 + 綜合系統性修正清單',
  phases: [
    { title: 'Review', detail: '每品 2 視角(完美主義/務實電商客戶)平行審查' },
    { title: 'Synthesize', detail: '綜合系統性瑕疵 + 排序修正' },
  ],
}

// args = [{ name, paths:[...], mode, name_guess }]（容錯：若被當字串傳入則 parse）
const PRODUCTS = (typeof args === 'string' ? JSON.parse(args) : args) || []

const SHOT_SCHEMA = {
  type: 'object',
  required: ['client_accept', 'score', 'verdict', 'flaws'],
  properties: {
    client_accept: { type: 'boolean', description: '真實電商客戶會不會接受這張登商品頁(務實標準，非吹毛求疵)' },
    score: { type: 'integer', description: '1-10' },
    verdict: { type: 'string', description: '一句總評(繁體中文)' },
    flaws: { type: 'array', items: { type: 'string' }, description: '具體瑕疵，從最嚴重排(繁體中文)；沒有就空陣列' },
    worst_flaw_type: { type: 'string', description: 'doubling|floating|seam|distortion|gibberish_text|texture_bleed|bad_hand|oversized|lighting|none|other' },
  },
}
const REVIEW_SCHEMA = {
  type: 'object',
  required: ['name', 'shots', 'root_cause', 'fix_recommendation'],
  properties: {
    name: { type: 'string' },
    shots: { type: 'array', items: SHOT_SCHEMA },
    root_cause: { type: 'string', description: '若有瑕疵，推測的根因(繁體中文)' },
    fix_recommendation: { type: 'string', description: '最該動哪裡(繁體中文)' },
  },
}

function reviewPrompt(p, lens) {
  const imgs = p.paths.map((x, i) => `第${i + 1}張: ${x}`).join('\n')
  const lensText = lens === 'strict'
    ? '你是極度挑剔的廣告攝影總監，標準是「頂尖品牌首頁大圖」。'
    : '你是務實的電商賣家客戶：圖只要乾淨、專業、沒有明顯 AI 破綻就會採用，不要求藝術完美。'
  return `${lensText}
這是商品「${p.name_guess || p.name}」的 AI 生成成品(模式=${p.mode})。請用 Read 工具逐一打開下列圖片**親眼看**，再評：
${imgs}

判斷標準(電商可用)：產品本體不可變形/比例怪；不可有雙重或第二個產品、拼接平鋪、亂碼假字、
浮空不接地、明顯接縫、材質滲入背景、壞掉的手；穿戴/手持要自然。乾淨專業即可接受。
對每張圖回一個 shot 物件(順序對應)。client_accept=務實客戶會不會用。誠實、繁體中文、不要簡體。`
}

// Review：每品兩視角平行審
const reviews = await pipeline(
  PRODUCTS,
  async (p) => {
    const [strict, practical] = await parallel([
      () => agent(reviewPrompt(p, 'strict'), { label: `strict:${p.name}`, phase: 'Review', schema: REVIEW_SCHEMA }),
      () => agent(reviewPrompt(p, 'practical'), { label: `client:${p.name}`, phase: 'Review', schema: REVIEW_SCHEMA }),
    ])
    return { name: p.name, mode: p.mode, strict, practical }
  }
)

// 綜合：以「務實客戶」為及格線(避免 harsh 過嚴)，但納入完美主義者抓到的具體破綻
const SYNTH_SCHEMA = {
  type: 'object',
  required: ['accepted', 'rejected', 'systematic_issues', 'priority_fixes'],
  properties: {
    accepted: { type: 'array', items: { type: 'string' }, description: '務實客戶可接受的產品名' },
    rejected: { type: 'array', items: { type: 'string' }, description: '仍不可接受的產品名' },
    systematic_issues: {
      type: 'array',
      items: {
        type: 'object',
        required: ['issue', 'affected', 'fix_approach'],
        properties: {
          issue: { type: 'string', description: '系統性瑕疵描述(繁體中文)' },
          affected: { type: 'array', items: { type: 'string' } },
          fix_approach: { type: 'string', description: '建議修法(繁體中文，越具體越好)' },
        },
      },
    },
    priority_fixes: { type: 'array', items: { type: 'string' }, description: '依影響力排序的修正動作(繁體中文)' },
  },
}
const synthesis = await agent(
  `以下是 SnapStudio 各展示品的雙視角審查結果(JSON)。請綜合：
- 及格線用「務實電商客戶 client_accept」，但要把完美主義者抓到的**具體可修破綻**納入系統性問題。
- 找出跨產品的**系統性瑕疵**(同一種破綻出現在多個產品)，按影響力排序修正動作。
審查資料：
${JSON.stringify(reviews, null, 2)}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { reviews, synthesis }
