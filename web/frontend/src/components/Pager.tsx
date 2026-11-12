import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { Page } from './usePage'
import './pager.css'


/** Under a table of more than one page: the rows shown, Previous and Next. Nothing for a table
 * of one page. */
export function Pager<T>({ page, noun }: { page: Page<T>; noun: string }) {
  if (page.total <= page.size) return null
  const from = page.at * page.size + 1
  const to = Math.min(page.total, from + page.size - 1)
  const pages = Math.ceil(page.total / page.size)
  return (
    <nav className="pager" aria-label="Pages">
      <span className="pager-range">{`${noun} ${from} to ${to} of ${page.total.toLocaleString('en-US')}`}</span>
      <button type="button" disabled={page.at === 0} onClick={() => page.go(page.at - 1)}>
        <ChevronLeft size={14} aria-hidden /> Previous
      </button>
      <button type="button" disabled={page.at >= pages - 1} onClick={() => page.go(page.at + 1)}>
        Next <ChevronRight size={14} aria-hidden />
      </button>
    </nav>
  )
}
