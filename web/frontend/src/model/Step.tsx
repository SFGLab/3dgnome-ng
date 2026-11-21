import type { ReactNode } from 'react'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

/** One numbered step of the form. */
export function Step({ n, title, note, children }: { n: number; title: string; note?: string; children: ReactNode }) {
  return (
    <section className="step">
      <h2 className={cn('step-head')}>
        <span className={cn('step-number')}>{n}</span>
        {title}
        {note && <span className={cn('step-note')}>{note}</span>}
      </h2>
      {children}
    </section>
  )
}
