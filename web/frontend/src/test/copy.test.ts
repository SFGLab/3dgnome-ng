import { describe, expect, it } from 'vitest'
import { checkCopy } from './copy'

function html(s: string): HTMLElement {
  const d = document.createElement('div')
  d.innerHTML = s
  return d
}

describe('checkCopy', () => {
  it('flags a heading that opens with a wh-word', () => {
    expect(checkCopy(html('<h2>How it works</h2>'))).toHaveLength(1)
  })
  it('flags a sentence that opens with a wh-word', () => {
    expect(checkCopy(html('<p>The model runs. When it ends you get files.</p>'))).toHaveLength(1)
  })
  it('flags a so clause', () => {
    expect(checkCopy(html('<p>It runs, so it is fast.</p>'))).toHaveLength(1)
  })
  it('flags a because clause', () => {
    expect(checkCopy(html('<p>It is fast because it batches.</p>'))).toHaveLength(1)
  })
  it('passes plain text', () => {
    expect(checkCopy(html('<h2>Distance law</h2><p>The model has loops. The form sets the loops.</p>'))).toEqual([])
  })
})
