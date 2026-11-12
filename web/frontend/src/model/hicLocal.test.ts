import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readHicWindow } from './hicLocal'

const meta = { chromosomes: [{ name: 'All', size: 1 }, { name: '1', size: 248_956_422 }], resolutions: [2_500_000, 100_000, 25_000, 5_000] }
const calls: unknown[][] = []

vi.mock('hic-straw', () => ({
  default: class {
    constructor(readonly args: unknown) {}
    async getMetaData() {
      return meta
    }
    async getContactRecords(...a: unknown[]) {
      calls.push(a)
      return [
        { bin1: 460, bin2: 460, counts: 61 },
        { bin1: 461, bin2: 460, counts: 12.6 },
        { bin1: 462, bin2: 470, counts: 0 },
      ]
    }
  },
}))

const REGION = { chrom: 'chr1', start: 11_500_000, end: 12_600_000 }

describe('readHicWindow', () => {
  beforeEach(() => {
    calls.length = 0
  })

  it('writes the window as BEDPE lines with whole counts', async () => {
    const text = await readHicWindow(new File(['x'], 'mine.hic'), REGION, 25_000)
    expect(text.trim().split('\n')).toEqual([
      'chr1\t11500000\t11525000\tchr1\t11500000\t11525000\t61',
      'chr1\t11500000\t11525000\tchr1\t11525000\t11550000\t13',
    ])
  })

  it('asks for the chromosome by the name the file uses', async () => {
    await readHicWindow(new File(['x'], 'mine.hic'), REGION, 25_000)
    expect(calls[0]).toEqual(['NONE', { chr: '1', start: 11_500_000, end: 12_600_000 }, { chr: '1', start: 11_500_000, end: 12_600_000 }, 'BP', 25_000])
  })

  it('names the resolutions of a file without the one asked for', async () => {
    await expect(readHicWindow(new File(['x'], 'mine.hic'), REGION, 10_000)).rejects.toThrow('mine.hic has no 10 kb resolution. It has 5 kb, 25 kb, 100 kb and 2.5 Mb.')
  })

  it('names a chromosome the file lacks', async () => {
    await expect(readHicWindow(new File(['x'], 'mine.hic'), { ...REGION, chrom: 'chr2' }, 25_000)).rejects.toThrow('mine.hic has no chromosome chr2.')
  })
})
