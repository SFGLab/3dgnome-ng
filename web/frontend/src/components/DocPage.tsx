import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import '../styles/doc.css'

/**
 * A page of the site that is a document. It has the list of its sections on the side, its title and
 * its lead, then its sections.
 * @param sections the id of a section of the page, or the path of another page, each with its label
 * @param wide a reading column for tables and figures
 */
export function DocPage({
  title,
  lead,
  sections,
  wide = false,
  children,
}: {
  title: string
  lead: ReactNode
  sections: readonly (readonly [to: string, label: string])[]
  wide?: boolean
  children: ReactNode
}) {
  return (
    <div className="doc-page">
      <aside className="side-panel">
        <p className="eyebrow">On this page</p>
        {sections.map(([to, label]) =>
          to.startsWith('/') ? (
            <Link key={to} to={to}>
              {label}
            </Link>
          ) : (
            <a key={to} href={to.startsWith('#') ? to : `#${to}`}>
              {label}
            </a>
          ),
        )}
      </aside>
      <div className={wide ? 'reading wide' : 'reading'}>
        <header className="page-head">
          <h1>{title}</h1>
          <p className="lead">{lead}</p>
        </header>
        {children}
      </div>
    </div>
  )
}
