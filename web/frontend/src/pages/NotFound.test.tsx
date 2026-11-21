import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { ROUTES } from '../routes'
import { checkCopy } from '../test/copy'
import { walk } from './chainWalk'
import { NotFound } from './NotFound'

const renderPage = () =>
  render(
    <MemoryRouter>
      <NotFound />
    </MemoryRouter>,
  )

/** Returns a fixed sequence in place of the random numbers, the same chain every time. */
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
    expect(screen.getByText('404')).toHaveClass('not-found-code')
    expect(screen.getByRole('link', { name: /Model a real region/ })).toHaveAttribute('href', ROUTES.model)
    expect(screen.getByRole('link', { name: 'Open an example' })).toHaveAttribute('href', ROUTES.examples)
    expect(screen.getByRole('link', { name: 'My jobs' })).toHaveAttribute('href', ROUTES.jobs)
  })

  it('draws a chain of 404 steps and folds another one on request', () => {
    renderPage()
    const points = () => screen.getByRole('img', { name: /A chain of 404 steps folded at random/ }).querySelector('polyline')!.getAttribute('points')!
    const first = points()
    expect(first.split(' ')).toHaveLength(405)
    expect(screen.getByText('Structure 1 of an ensemble that fits no data')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Fold it again/ }))
    expect(screen.getByText('Structure 2 of an ensemble that fits no data')).toBeInTheDocument()
    expect(points()).not.toBe(first)
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
