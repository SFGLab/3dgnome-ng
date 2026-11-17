/** Checks page text against design/writing-rules.md. Returns one message per violation. */
const WH = /^(What|Where|How|Who|Whether|When|Which|Why)\b/
const BLOCKS = 'h1,h2,h3,h4,h5,h6,p,li,td,th,dt,dd,figcaption,label,button,a,summary'

export function checkCopy(root: HTMLElement): string[] {
  const found = new Set<string>()
  for (const el of root.querySelectorAll<HTMLElement>(BLOCKS)) {
    const text = (el.textContent ?? '').replace(/\s+/g, ' ').trim()
    if (!text) continue
    for (const sentence of text.split(/(?<=[.!?])\s+/)) {
      if (WH.test(sentence)) found.add(`opens with a wh-word: "${sentence}"`)
    }
    if (text.includes(', so ')) found.add(`has a ", so" clause: "${text}"`)
    if (/\bbecause\b/i.test(text)) found.add(`has a "because" clause: "${text}"`)
  }
  return [...found]
}
