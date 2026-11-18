import type { ReactNode } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { ROUTES } from '../routes'
import { CiteStrip } from './CiteStrip'
import { PrivacyNotice } from './PrivacyNotice'
import { ThemeSwitch } from './ThemeSwitch'
import { Wordmark } from './Wordmark'
import './layout.css'

const NAV: [string, string][] = [
  ['Model a region', ROUTES.model],
  ['My jobs', ROUTES.jobs],
  ['Method', ROUTES.method],
  ['Results', ROUTES.results],
  ['Help', ROUTES.help],
  ['Cite', ROUTES.cite],
]

/** The top bar, the page, the cite strip and the privacy notice of a first visit. */
export function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="layout">
      <header className="top-bar">
        <Link className="brand" to={ROUTES.title}>
          <Wordmark />
        </Link>
        <nav>
          {NAV.map(([label, to]) => (
            <NavLink key={to} to={to}>
              {label}
            </NavLink>
          ))}
        </nav>
        <ThemeSwitch />
      </header>
      <main>{children}</main>
      <CiteStrip />
      <PrivacyNotice />
    </div>
  )
}
