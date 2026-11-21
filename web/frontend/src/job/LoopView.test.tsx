import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { Loop } from './api'
import { LoopView } from './LoopView'
import { describeLoop, sameLoop, ticks, zoomed } from './loops'
import { wheelFactor } from './wheel'

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
    // A press and a release on a loop with no drag between them select it. The view takes the
    // pointer for a drag, so it reads the loop from the press and not from a click.
    const press = (name: string) => {
      fireEvent.pointerDown(screen.getByRole('button', { name }))
      fireEvent.pointerUp(screen.getByRole('button', { name }))
    }
    press(describeLoop(LOOPS[1]))
    expect(onSelect).toHaveBeenLastCalledWith(1)
    rerender(<LoopView chrom="chr1" region={[900_000, 2_000_000]} loops={LOOPS} onSelect={onSelect} selected={1} hint="the hint" />)
    // The line under the view names the selected loop.
    expect(screen.getByText(`Selected: ${describeLoop(LOOPS[1])}`)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: describeLoop(LOOPS[1]) })).toHaveAttribute('aria-pressed', 'true')
    press(describeLoop(LOOPS[1]))
    expect(onSelect).toHaveBeenLastCalledWith(null)
  })

  it('marks the selected loop: its arc, its two anchors and the others stepped back', () => {
    const onSelect = vi.fn()
    const { container } = render(<LoopView chrom="chr1" region={[900_000, 2_000_000]} loops={LOOPS} onSelect={onSelect} selected={0} />)
    expect(container.querySelectorAll('.loop-arc.selected')).toHaveLength(1)
    expect(container.querySelectorAll('.loop-arc.dimmed')).toHaveLength(1)
    expect([...container.querySelectorAll('.loop-anchor')].map((m) => m.textContent)).toEqual(['A', 'B'])
    fireEvent.click(screen.getByRole('button', { name: 'Clear the selected loop' }))
    expect(onSelect).toHaveBeenCalledWith(null)
  })

  it('still selects a loop where a drag is also read, and a drag selects none', () => {
    const [onSelect, onWindow] = [vi.fn(), vi.fn()]
    const { container } = render(<LoopView chrom="chr1" region={[1_000_000, 2_000_000]} loops={LOOPS} onSelect={onSelect} onWindow={onWindow} />)
    const svg = container.querySelector('.loop-view-arcs')!
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 500, height: 72, right: 500, bottom: 72, x: 0, y: 0, toJSON: () => ({}) })
    const hit = screen.getByRole('button', { name: describeLoop(LOOPS[0]) })
    fireEvent(hit, new MouseEvent('pointerdown', { clientX: 100, bubbles: true }))
    fireEvent(svg, new MouseEvent('pointerup', { clientX: 100, bubbles: true }))
    expect(onSelect).toHaveBeenLastCalledWith(0)
    expect(onWindow).not.toHaveBeenCalled()
    fireEvent(hit, new MouseEvent('pointerdown', { clientX: 100, bubbles: true }))
    fireEvent(svg, new MouseEvent('pointermove', { clientX: 300, bubbles: true }))
    fireEvent(svg, new MouseEvent('pointerup', { clientX: 300, bubbles: true }))
    expect(onWindow).toHaveBeenCalledWith(1_200_000, 1_600_000)
    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it('scales the part in use about the pointer on the wheel and offers the way back', () => {
    expect(zoomed([1_000_000, 2_000_000], [1_000_000, 2_000_000], 1_500_000, 0.8)).toEqual([1_100_000, 1_900_000])
    // Out of a part the zoom stops at the whole region.
    expect(zoomed([1_100_000, 1_900_000], [1_000_000, 2_000_000], 1_500_000, 2)).toEqual([1_000_000, 2_000_000])
    // The position under the pointer keeps its place: a quarter into the part before and after.
    expect(zoomed([1_000_000, 2_000_000], [0, 5_000_000], 1_250_000, 0.5)).toEqual([1_125_000, 1_625_000])
    const [onWindow, onReset] = [vi.fn(), vi.fn()]
    const { container } = render(<LoopView chrom="chr1" region={[1_200_000, 1_600_000]} bounds={[1_000_000, 2_000_000]} loops={LOOPS} onWindow={onWindow} onReset={onReset} />)
    const svg = container.querySelector('.loop-view-arcs')!
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 72, right: 400, bottom: 72, x: 0, y: 0, toJSON: () => ({}) })
    // One notch of a mouse wheel takes about a tenth off the part, the same on either side of the pointer.
    fireEvent(svg, new WheelEvent('wheel', { deltaY: -100, clientX: 200, bubbles: true, cancelable: true }))
    const [a, b] = onWindow.mock.lastCall!
    expect(b - a).toBeGreaterThan(0.88 * 400_000)
    expect(b - a).toBeLessThan(0.93 * 400_000)
    expect(Math.abs(a + b - 2_800_000)).toBeLessThanOrEqual(1)
    // A small step of a trackpad scales by well under a hundredth.
    expect(wheelFactor({ deltaY: -4 })).toBeGreaterThan(0.99)
    expect(wheelFactor({ deltaY: 4 })).toBeLessThan(1.01)
    // A step in lines counts as its pixels and no event counts for more than a notch.
    expect(wheelFactor({ deltaY: 3, deltaMode: 1 })).toBeCloseTo(wheelFactor({ deltaY: 48 }))
    expect(wheelFactor({ deltaY: 5000 })).toBe(wheelFactor({ deltaY: 100 }))
    expect(wheelFactor({ deltaY: 100 }) * wheelFactor({ deltaY: -100 })).toBeCloseTo(1)
    fireEvent.click(screen.getByRole('button', { name: 'Whole region' }))
    expect(onReset).toHaveBeenCalled()
  })

  it('draws the part another view shows and reports a drag as a part of the region', () => {
    const onWindow = vi.fn()
    const { container } = render(<LoopView chrom="chr1" region={[1_000_000, 2_000_000]} loops={LOOPS} window={[1_250_000, 1_500_000]} onWindow={onWindow} />)
    const box = container.querySelector('.loop-window')!
    expect(box.getAttribute('x')).toBe('250')
    expect(box.getAttribute('width')).toBe('250')
    const svg = container.querySelector('.loop-view-arcs')!
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
