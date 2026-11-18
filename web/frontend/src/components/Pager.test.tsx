import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Pager } from './Pager'
import { PAGE, PAGE_SIZES, usePage } from './usePage'

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
  beforeEach(() => localStorage.clear())

  it('shows no pager for a table that fits the smallest page', () => {
    render(<List n={PAGE_SIZES[0]} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(PAGE_SIZES[0])
    expect(screen.queryByRole('navigation', { name: 'Pages' })).toBeNull()
  })

  it('offers the rows per page and starts again from the first row', () => {
    render(<List n={120} />)
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.change(screen.getByLabelText('Rows per page'), { target: { value: '100' } })
    expect(screen.getAllByRole('listitem')).toHaveLength(100)
    expect(screen.getByText('rows 1 to 100 of 120')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Rows per page'), { target: { value: '250' } })
    expect(screen.getAllByRole('listitem')).toHaveLength(120)
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()
  })

  it('keeps the chosen rows per page for the next table', () => {
    const first = render(<List n={120} />)
    fireEvent.change(screen.getByLabelText('Rows per page'), { target: { value: '25' } })
    first.unmount()
    render(<List n={120} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(25)
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
