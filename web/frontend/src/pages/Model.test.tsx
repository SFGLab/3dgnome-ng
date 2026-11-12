import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { Model } from './Model'

function renderModel(path = '/model') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Model />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('Model a region page', () => {
  beforeEach(() => {
    mockApi()
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it('has the five steps in the design order', async () => {
    renderModel()
    const steps = await screen.findAllByRole('heading', { level: 2 })
    expect(steps.map((h) => h.textContent)).toEqual([
      '1Cell type and region',
      '2Loops',
      '3Structural variantsoptional',
      '4Contact background',
      '5Ensemble and simulation settings',
    ])
  })

  it('lists the cell types from the catalog with their data', async () => {
    renderModel()
    const card = await screen.findByRole('radio', { name: /GM12878/ })
    expect(within(card).getByText('lymphoblastoid')).toBeInTheDocument()
    expect(within(card).getByText('enhancers')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Your own data/ })).toBeDisabled()
  })

  it('resolves a gene name to its region', async () => {
    renderModel()
    fireEvent.change(await screen.findByLabelText('Region, chr:start-end or a gene name'), { target: { value: 'TNFRSF8' } })
    expect(await screen.findByText('chr1:11,500,000-12,600,000, 1.1 Mb')).toBeInTheDocument()
  })

  it('shows the error of a region it cannot resolve', async () => {
    renderModel()
    fireEvent.change(await screen.findByLabelText('Region, chr:start-end or a gene name'), { target: { value: 'nowhere' } })
    expect(await screen.findByText(/is neither chr:start-end nor a known gene name/)).toBeInTheDocument()
  })

  it('fills the form from an example', async () => {
    renderModel()
    fireEvent.click(await screen.findByRole('button', { name: /MYC/ }))
    expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('chr8:127,500,000-129,000,000')
    expect(screen.getByRole('radio', { name: /GM12878/ })).toBeChecked()
  })

  it('opens filled from an example link', async () => {
    renderModel('/model?example=TNFRSF8')
    await waitFor(() => expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('TNFRSF8'))
  })

  it('lists the recent jobs of this browser', async () => {
    rememberJob({ url: '/jobs/abc12345', label: 'GM12878 chr1:11.5-12.6 Mb', name: 'TNFRSF8 run', at: 1 })
    renderModel()
    const link = await screen.findByRole('link', { name: /TNFRSF8 run/ })
    expect(link).toHaveAttribute('href', '/jobs/abc12345')
    expect(await within(link).findByText('done')).toBeInTheDocument()
  })

  it('keeps to the writing rules', async () => {
    const { container } = renderModel()
    await screen.findByRole('radio', { name: /GM12878/ })
    expect(checkCopy(container)).toEqual([])
  })
})
