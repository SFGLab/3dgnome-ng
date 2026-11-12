import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const SRC = join(dirname(fileURLToPath(import.meta.url)), '..')

function cssFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? cssFiles(join(dir, e.name)) : e.name.endsWith('.css') ? [join(dir, e.name)] : [],
  )
}

describe('stylesheets', () => {
  it('balance their braces, a block never closes before it opens', () => {
    for (const f of cssFiles(SRC)) {
      let depth = 0
      for (const ch of readFileSync(f, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')) {
        if (ch === '{') depth++
        if (ch === '}') depth--
        expect(depth, f).toBeGreaterThanOrEqual(0)
      }
      expect(depth, f).toBe(0)
    }
  })
})
