import type { ReactNode } from 'react'

/** One numbered step of the form. */
export function Step({ n, title, note, children }: { n: number; title: string; note?: string; children: ReactNode }) {
  return (
    <section className="step">
      <h2 className="step-head">
        <span className="step-number">{n}</span>
        {title}
        {note && <span className="step-note">{note}</span>}
      </h2>
      {children}
    </section>
  )
}
