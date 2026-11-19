import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { RegionField } from './RegionField'

function Field({ start = '' }: { start?: string }) {
  const [value, setValue] = useState(start)
  return <RegionField cell="GM12878" value={value} onChange={setValue} />
}

function renderField(start = '') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <Field start={start} />
    </QueryClientProvider>,
  )
  return screen.getByLabelText('Region, chr:start-end or a gene name')
}

describe('RegionField', () => {
  beforeEach(() => {
    mockApi()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('lists the genes that match the typed letters, each with its position', async () => {
    const field = renderField()
    fireEvent.focus(field)
    fireEvent.change(field, { target: { value: 'my' } })
    const options = await screen.findAllByRole('option')
    expect(options.map((o) => o.textContent)).toEqual(['MYCchr8:127,735,433', 'MYCBPchr1:38,873,368'])
    expect(field).toHaveAttribute('aria-expanded', 'true')
  })

  it('puts a clicked gene in the field and closes the list', async () => {
    const field = renderField()
    fireEvent.focus(field)
    fireEvent.change(field, { target: { value: 'my' } })
    fireEvent.mouseDown(await screen.findByRole('option', { name: /MYCBP/ }))
    expect(field).toHaveValue('MYCBP')
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('takes a gene with the arrow keys and Enter', async () => {
    const field = renderField()
    fireEvent.focus(field)
    fireEvent.change(field, { target: { value: 'my' } })
    await screen.findAllByRole('option')
    fireEvent.keyDown(field, { key: 'ArrowDown' })
    expect(screen.getByRole('option', { name: /MYCBP/ })).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(field, { key: 'Enter' })
    expect(field).toHaveValue('MYCBP')
  })

  it('lists nothing for coordinates, for one letter or for a field that is not in use', async () => {
    const field = renderField('TNF')
    expect(screen.queryByRole('listbox')).toBeNull()
    fireEvent.focus(field)
    fireEvent.change(field, { target: { value: 'chr8:127,000,000-128,500,000' } })
    await waitFor(() => expect(screen.getByText('chr8:127,500,000-129,000,000, 1.5 Mb')).toBeInTheDocument())
    expect(screen.queryByRole('listbox')).toBeNull()
    fireEvent.change(field, { target: { value: 'm' } })
    await new Promise((r) => setTimeout(r, 250))
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('does not list the one gene that is in the field already', async () => {
    const field = renderField()
    fireEvent.focus(field)
    fireEvent.change(field, { target: { value: 'TNFRSF8' } })
    await new Promise((r) => setTimeout(r, 250))
    expect(screen.queryByRole('listbox')).toBeNull()
    fireEvent.keyDown(field, { key: 'Enter' })
    await waitFor(() => expect(screen.getByText('chr1:11,500,000-12,600,000, 1.1 Mb')).toBeInTheDocument())
  })

  it('gives the form a gene name once it is chosen, entered or the field is left, and coordinates at once', async () => {
    const seen: string[] = []
    function Spy() {
      const [value, setValue] = useState('')
      return <RegionField cell="GM12878" value={value} onChange={(v) => { seen.push(v); setValue(v) }} />
    }
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <Spy />
      </QueryClientProvider>,
    )
    const field = screen.getByLabelText('Region, chr:start-end or a gene name')
    fireEvent.focus(field)
    fireEvent.change(field, { target: { value: 'MY' } })
    fireEvent.change(field, { target: { value: 'MYC' } })
    expect(seen).toEqual([])
    fireEvent.blur(field)
    expect(seen).toEqual(['MYC'])
    fireEvent.change(field, { target: { value: 'chr8:1-2' } })
    expect(seen).toEqual(['MYC', 'chr8:1-2'])
  })
})
