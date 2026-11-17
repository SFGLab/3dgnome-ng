import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Privacy } from './Privacy'

function renderPage() {
  return render(
    <MemoryRouter>
      <Privacy />
    </MemoryRouter>,
  )
}

describe('Privacy page', () => {
  it('says what the site does not do', () => {
    renderPage()
    expect(screen.getByText('The site sets no cookies and does no tracking or analytics.')).toBeInTheDocument()
    expect(screen.getByText('It asks for no name and no e-mail address, and it has no accounts.')).toBeInTheDocument()
  })

  it('lists what the server keeps and for how long', () => {
    renderPage()
    const rows = screen.getAllByRole('row').slice(1)
    expect(rows.map((r) => r.querySelector('td')!.textContent)).toEqual(['Job requests', 'Uploaded files and their jobs', 'Synced job lists', 'Links to contact maps', 'Server log'])
    expect(screen.getByText('14 days, then deleted')).toBeInTheDocument()
  })

  it('names the controller, the contact and the authority', () => {
    renderPage()
    expect(screen.getByText(/Warsaw University of Technology, Politechnika Warszawska, Pl\. Politechniki 1, 00-661 Warszawa/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'iod@pw.edu.pl' })).toHaveAttribute('href', 'mailto:iod@pw.edu.pl')
    expect(screen.getByText(/President of the Personal Data Protection Office of Poland/)).toBeInTheDocument()
  })

  it('warns about genetic data', () => {
    renderPage()
    expect(screen.getByRole('heading', { name: 'Genetic data' })).toBeInTheDocument()
    expect(screen.getByText(/special category under the GDPR/)).toBeInTheDocument()
  })

  it('follows the writing rules', () => {
    const { container } = renderPage()
    expect(checkCopy(container)).toEqual([])
  })
})
