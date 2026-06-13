export const meta = {
  name: 'snapstudio-source-finder',
  description: '多 agent 上網找乾淨高解析商品來源照並下載驗證',
  phases: [{ title: 'Source', detail: '每個目標一個 agent：搜尋→下載→PIL 驗證' }],
}

// args = [{ name, query }]
const TARGETS = (typeof args === 'string' ? JSON.parse(args) : args) || []
const DIR = '/workspace/Deep_Generative_Model/HW7_snapstudio/examples/products/sourced'

const SCHEMA = {
  type: 'object',
  required: ['name', 'path', 'ok', 'note'],
  properties: {
    name: { type: 'string' },
    path: { type: 'string', description: '成功下載的絕對路徑；失敗給空字串' },
    ok: { type: 'boolean' },
    width: { type: 'integer' }, height: { type: 'integer' },
    source_url: { type: 'string' },
    note: { type: 'string', description: '繁體中文，找了什麼、為何選它/為何失敗' },
  },
}

const results = await parallel(TARGETS.map((t) => () => agent(
  `你的任務：找一張「${t.query}」的**乾淨、高解析、單一商品**照片並下載到本機，供商品攝影 pipeline 去背使用。
嚴格步驟：
1) 用 WebSearch 搜尋(關鍵字加上 "product photo, plain background, high resolution")，找候選圖片。
2) 找出**直接圖片 URL**(結尾 .jpg/.jpeg/.png；優先 images.unsplash.com、cdn、raw.githubusercontent 這類可靠且允許下載的來源；避開 Wikimedia upload(會回錯誤頁) 與需登入的站)。可先 WebFetch 該頁找出真正的圖片直連。
3) 先 \`mkdir -p ${DIR}\`，再用 Bash 下載：
   curl -sL -A "Mozilla/5.0" -o ${DIR}/${t.name}.jpg "<圖片URL>"
4) 用 Bash 驗證是真圖且夠大：
   /workspace/.venv-1/bin/python -c "from PIL import Image;im=Image.open('${DIR}/${t.name}.jpg');print(im.size)"
   寬與高都必須 > 500；若失敗(HTML錯誤頁/太小)就換下一個 URL，最多試 6 個。
5) 用 Read 親眼確認那張圖：必須是**單一、完整、清晰、背景乾淨**的「${t.query}」，不是多件、不是被裁切、不是插畫。不合格就換。
回報 {name:"${t.name}", path, ok, width, height, source_url, note}。務必繁體中文 note。`,
  { label: `source:${t.name}`, phase: 'Source', schema: SCHEMA, agentType: 'general-purpose' }
)))

return results.filter(Boolean)
