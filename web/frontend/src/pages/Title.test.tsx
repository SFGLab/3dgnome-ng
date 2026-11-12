import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { Layout } from '../components/Layout'
import { ROUTES } from '../routes'
import { checkCopy } from '../test/copy'
import { Title } from './Title'

function renderTitle() {
  return render(
    <MemoryRouter>
      <Layout>
        <Title />
      </Layout>
    </MemoryRouter>,
  )
}

describe('Title page', () => {
  it('has the name as its main heading and the lead', () => {
    renderTitle()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('3D-GNOMENG')
    expect(screen.getByText(/computes 3D structures of genomic regions up to 20 Mb/)).toBeInTheDocument()
  })

  it('links to the form and the results', () => {
    renderTitle()
    const model = screen.getAllByRole('link', { name: /Model a region/ }).map((a) => a.getAttribute('href'))
    expect(model).toContain(ROUTES.model)
    expect(screen.queryByRole('link', { name: /Comparison with Hi-C and expression data/ })).toBeNull()
  })

  it('links to the previous services', () => {
    renderTitle()
    expect(screen.getByRole('link', { name: /Open 3D-GNOME 3.0/ })).toHaveAttribute('href', 'https://3dgnome.mini.pw.edu.pl/old')
    expect(screen.getByRole('link', { name: /Open enhancer3D/ })).toHaveAttribute('href', 'https://3dgnome.mini.pw.edu.pl/old/enhancer3D/')
  })

  it('shows three examples', () => {
    renderTitle()
    for (const t of ['TNFRSF8, CTCF and RNAPII loops', 'The same locus, CTCF loops alone', 'Contacts against Hi-C']) {
      expect(screen.getByText(t)).toBeInTheDocument()
    }
  })

  it('keeps to the writing rules', () => {
    const { container } = renderTitle()
    expect(checkCopy(container)).toEqual([])
  })
})
