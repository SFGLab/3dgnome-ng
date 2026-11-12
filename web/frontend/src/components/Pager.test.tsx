import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Pager } from './Pager'
import { PAGE, usePage } from './usePage'

function List({ n, reset = 0 }: { n: number; reset?: number }) {
  const rows = Array.from({ length: n }, (_, i) => `row ${i + 1}`)
  const page = usePage(rows, reset)
  return (
    <div>
      <ul>
        {page.rows.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
      <Pager page={page} noun="rows" />
    </div>
  )
}

describe('Pager', () => {
  it('shows no pager for a page or less', () => {
    render(<List n={PAGE} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(PAGE)
    expect(screen.queryByRole('navigation', { name: 'Pages' })).toBeNull()
  })

  it('pages through more rows with the range shown', () => {
    const { container } = render(<List n={120} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(PAGE)
    expect(screen.getByText('rows 1 to 50 of 120')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByText('rows 101 to 120 of 120')).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')[0]).toHaveTextContent('row 101')
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()
    expect(checkCopy(container)).toEqual([])
  })

  it('goes back to the first page when the rows are filtered or sorted', () => {
    const { rerender } = render(<List n={120} />)
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    rerender(<List n={120} reset={1} />)
    expect(screen.getByText('rows 1 to 50 of 120')).toBeInTheDocument()
  })
})
