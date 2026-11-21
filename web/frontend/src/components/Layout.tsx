import type { ReactNode } from 'react'
import { Link, NavLink } from 'react-router-dom'
import { hidden } from '../flags'
import { ROUTES } from '../routes'
import { CiteStrip } from './CiteStrip'
import { PrivacyNotice } from './PrivacyNotice'
import { ThemeSwitch } from './ThemeSwitch'
import { Wordmark } from './Wordmark'
import { scoped } from '../lib/css'
import sheet from './layout.module.css'

const cn = scoped(sheet)

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
    <div className={cn('layout')}>
      <header className={cn('top-bar')}>
        <Link className="brand" to={ROUTES.title}>
          <Wordmark />
        </Link>
        <nav>
          {NAV.filter(([, to]) => !(to === ROUTES.method && hidden('method')) && !(to === ROUTES.results && hidden('results'))).map(([label, to]) => (
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
