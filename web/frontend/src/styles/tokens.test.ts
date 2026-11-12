import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const css = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'tokens.css'), 'utf8')

/** The custom property names declared in the first block that starts with `selector`. */
function names(selector: string): Set<string> {
  const start = css.indexOf(selector)
  if (start < 0) return new Set()
  const body = css.slice(css.indexOf('{', start) + 1, css.indexOf('}', start))
  return new Set([...body.matchAll(/(--[\w-]+)\s*:/g)].map((m) => m[1]))
}

describe('design tokens', () => {
  it('defines the same tokens for light and dark', () => {
    const light = names(':root {')
    const dark = names(':root[data-theme="dark"]')
    expect(light.size).toBeGreaterThan(0)
    expect([...dark].sort()).toEqual([...light].sort())
  })

  it('defines the tokens the pages use', () => {
    const light = names(':root {')
    for (const t of ['--accent', '--ink', '--muted', '--faint', '--line', '--surface', '--panel', '--bg', '--ctcf', '--rnapii', '--promoter', '--enhancer', '--font', '--mono']) {
      expect(light, t).toContain(t)
    }
  })

  it('repeats the dark tokens for the system preference', () => {
    expect(css).toContain('@media (prefers-color-scheme: dark)')
    expect(css).toContain(':root:not([data-theme="light"])')
  })
})
