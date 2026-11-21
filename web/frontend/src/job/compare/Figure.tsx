import { ChevronRight, Download } from 'lucide-react'
import { type ReactNode, useRef } from 'react'
import { savePng } from '../export/savePng'
import { scoped } from '../../lib/css'
import sheet from '../job.module.css'

const cn = scoped(sheet)

/** A figure of the comparison as one part of an accordion: its title opens and closes it, and
 * an open figure has a button that saves it as a PNG. */
export function Figure({ title, file, open, onToggle, children }: { title: string; file: string; open: boolean; onToggle: () => void; children: ReactNode }) {
  const box = useRef<HTMLElement>(null)
  const save = () => {
    // The plot of the figure, and not an icon of its head.
    const svg = box.current?.querySelector<SVGSVGElement>('svg.expression-plot')
    if (svg) void savePng(svg, title, file)
  }
  return (
    <figure className={cn(`compare-figure${open ? ' open' : ''}`)} ref={box}>
      <figcaption>
        <button type="button" className={cn('figure-head')} aria-expanded={open} onClick={onToggle}>
          <ChevronRight size={14} aria-hidden /> {title}
        </button>
        {open && (
          <button type="button" className={cn('figure-save')} aria-label={`Save the figure ${title} as a PNG`} onClick={save}>
            <Download size={12} aria-hidden /> PNG
          </button>
        )}
      </figcaption>
      {open && children}
    </figure>
  )
}
