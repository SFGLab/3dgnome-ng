import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { Loop } from './api'
import { describeLoop, LoopView, sameLoop, ticks } from './LoopView'

const LOOPS: Loop[] = [
  { factor: 'CTCF', chrom: 'chr1', a: [1_000_000, 1_002_000], b: [1_400_000, 1_402_000], span: 398_000, pets: 12 },
  { factor: 'RNAPII', chrom: 'chr1', a: [1_100_000, 1_101_000], b: [1_150_000, 1_151_000], span: 49_000, pets: 1 },
  { factor: 'CTCF', chrom: 'chr1', a: [5_000_000, 5_002_000], b: [5_400_000, 5_402_000], span: 398_000, pets: 4 },
]

describe('the loop view', () => {
  it('describes a loop by its factor, anchors, span and PETs', () => {
    expect(describeLoop(LOOPS[0])).toBe('CTCF loop, chr1:1,000,000-1,002,000 to chr1:1,400,000-1,402,000, 398 kb, 12 PETs')
    expect(describeLoop(LOOPS[1])).toMatch(/49 kb, 1 PET$/)
    expect(sameLoop(LOOPS[0], { ...LOOPS[0] })).toBe(true)
    expect(sameLoop(LOOPS[0], LOOPS[2])).toBe(false)
  })

  it('puts about five round ticks across a span', () => {
    expect(ticks(1_000_000, 2_000_000)).toEqual([1_000_000, 1_200_000, 1_400_000, 1_600_000, 1_800_000, 2_000_000])
    expect(ticks(20_030_000, 21_490_000)).toEqual([20_500_000, 21_000_000])
  })

  it('draws the loops that reach into the region and none beyond it', () => {
    render(<LoopView chrom="chr1" region={[900_000, 2_000_000]} loops={LOOPS} onSelect={vi.fn()} />)
    expect(screen.getAllByRole('button')).toHaveLength(2)
    expect(screen.getByRole('button', { name: describeLoop(LOOPS[0]) })).toBeInTheDocument()
  })

  it('reports a clicked loop by its place in the list and clears on a second click', () => {
    const onSelect = vi.fn()
    const { rerender } = render(<LoopView chrom="chr1" region={[900_000, 2_000_000]} loops={LOOPS} onSelect={onSelect} hint="the hint" />)
    expect(screen.getByText('the hint')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: describeLoop(LOOPS[1]) }))
    expect(onSelect).toHaveBeenLastCalledWith(1)
    rerender(<LoopView chrom="chr1" region={[900_000, 2_000_000]} loops={LOOPS} onSelect={onSelect} selected={1} hint="the hint" />)
    // The line under the view names the selected loop.
    expect(screen.getByText(describeLoop(LOOPS[1]))).toBeInTheDocument()
    expect(screen.getByRole('button', { name: describeLoop(LOOPS[1]) })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: describeLoop(LOOPS[1]) }))
    expect(onSelect).toHaveBeenLastCalledWith(null)
  })

  it('draws the part another view shows and reports a drag as a part of the region', () => {
    const onWindow = vi.fn()
    const { container } = render(<LoopView chrom="chr1" region={[1_000_000, 2_000_000]} loops={LOOPS} window={[1_250_000, 1_500_000]} onWindow={onWindow} />)
    const box = container.querySelector('.loop-window')!
    expect(box.getAttribute('x')).toBe('250')
    expect(box.getAttribute('width')).toBe('250')
    const svg = container.querySelector('svg')!
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 500, height: 72, right: 500, bottom: 72, x: 0, y: 0, toJSON: () => ({}) })
    // jsdom's pointer events carry no position, so mouse events of the same names stand in.
    const at = (type: string, clientX: number) => fireEvent(svg, new MouseEvent(type, { clientX, bubbles: true }))
    at('pointerdown', 100)
    at('pointermove', 300)
    expect(container.querySelector('.loop-brush')).toBeInTheDocument()
    fireEvent.pointerUp(svg)
    expect(onWindow).toHaveBeenCalledWith(1_200_000, 1_600_000)
    // A press that hardly moves is a click and asks for nothing.
    at('pointerdown', 100)
    at('pointermove', 102)
    fireEvent.pointerUp(svg)
    expect(onWindow).toHaveBeenCalledTimes(1)
  })

  it('marks a position inside the region', () => {
    const { container, rerender } = render(<LoopView chrom="chr1" region={[1_000_000, 2_000_000]} loops={LOOPS} cursor={1_500_000} />)
    expect(container.querySelector('.loop-cursor')?.getAttribute('x1')).toBe('500')
    rerender(<LoopView chrom="chr1" region={[1_000_000, 2_000_000]} loops={LOOPS} cursor={2_500_000} />)
    expect(container.querySelector('.loop-cursor')).not.toBeInTheDocument()
  })
})
