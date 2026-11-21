import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { ROUTES } from '../routes'
import { checkCopy } from '../test/copy'
import { fold404, HEIGHT, walk, WIDTH } from './chainWalk'
import { NotFound } from './NotFound'

const renderPage = () =>
  render(
    <MemoryRouter>
      <NotFound />
    </MemoryRouter>,
  )

/** A fixed sequence in place of the random numbers: the same chain every time. */
function fixed() {
  let k = 0
  return () => ((k++ * 37) % 100) / 100
}

describe('The page of an address the site does not have', () => {
  it('says the page is missing, in the words of its author, and links back into the site', () => {
    renderPage()
    expect(screen.getByRole('heading', { level: 1, name: 'No structure at this address' })).toBeInTheDocument()
    expect(
      screen.getByText('The link may be mistyped, or it was the link for a private job, that was deleted after 14 days. In its place the page has folded a chain of 404 steps for you using random data. Enjoy.'),
    ).toBeInTheDocument()
    expect(screen.getByText('- Nik')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Model a real region/ })).toHaveAttribute('href', ROUTES.model)
    expect(screen.getByRole('link', { name: 'Open an example' })).toHaveAttribute('href', ROUTES.examples)
    expect(screen.getByRole('link', { name: 'My jobs' })).toHaveAttribute('href', ROUTES.jobs)
  })

  it('has 404 as a sign folded from one chain, its digits the nearer strands and its two ends coloured', () => {
    renderPage()
    const sign = screen.getByRole('img', { name: '404, folded from one chain of beads' })
    expect(sign.querySelectorAll('path.strand.near').length).toBeGreaterThan(0)
    expect(sign.querySelectorAll('path.strand.far').length).toBeGreaterThan(0)
    expect(sign.querySelector('path.strand.end.promoter')).not.toBeNull()
    expect(sign.querySelector('path.strand.end.enhancer')).not.toBeNull()
  })

  it('draws a chain of 404 steps and folds another one, with another sign, on request', () => {
    renderPage()
    const points = () => screen.getByRole('img', { name: /A chain of 404 steps folded at random/ }).querySelector('polyline')!.getAttribute('points')!
    const sign = () => screen.getByRole('img', { name: '404, folded from one chain of beads' }).innerHTML
    const [first, firstSign] = [points(), sign()]
    expect(first.split(' ')).toHaveLength(405)
    expect(screen.getByText('Structure 1 of an ensemble that fits no data')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Fold it again/ }))
    expect(screen.getByText('Structure 2 of an ensemble that fits no data')).toBeInTheDocument()
    expect(points()).not.toBe(first)
    expect(sign()).not.toBe(firstSign)
  })

  it('makes the sign as one chain of three digits with a loose end either side, inside its drawing', () => {
    const beads = fold404(fixed())
    expect(beads.every((b) => b.x >= 0 && b.x <= WIDTH && b.y >= 0 && b.y <= HEIGHT)).toBe(true)
    // The beads of the digits come in three runs, with beads of the rest before, between and after them.
    const runs = beads.map((b) => (b.digit ? 'd' : 'r')).join('').match(/d+|r+/g)!
    expect(runs.map((r) => r[0])).toEqual(['r', 'd', 'r', 'd', 'r', 'd', 'r'])
    expect(runs.filter((r) => r[0] === 'd').every((r) => r.length > 50)).toBe(true)
    // A digit is nearer than the rest of the chain.
    expect(Math.min(...beads.filter((b) => b.digit).map((b) => b.layer))).toBeGreaterThan(Math.max(...beads.filter((b) => !b.digit).map((b) => b.layer)))
    // The same random numbers give the same chain.
    expect(fold404(fixed())).toEqual(beads)
  })

  it('keeps the random chain inside its drawing', () => {
    const chain = walk(404, fixed())
    expect(chain).toHaveLength(405)
    expect(chain.every(([x, y]) => x >= 0 && x <= 420 && y >= 0 && y <= 420)).toBe(true)
  })

  it('keeps to the writing rules', () => {
    const { container } = renderPage()
    expect(checkCopy(container)).toEqual([])
  })
})
