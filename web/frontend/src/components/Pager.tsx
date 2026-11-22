import { ChevronLeft, ChevronRight } from 'lucide-react'
import { PAGE_SIZES, type Page } from './usePage'
import { scoped } from '../lib/css'
import sheet from './pager.module.css'

const cn = scoped(sheet)

/**
 * The controls over a table, which are the rows shown, the rows per page, Previous and Next.
 * Nothing for a table that fits the smallest page.
 */
export function Pager<T>({ page, noun }: { page: Page<T>; noun: string }) {
  if (page.total <= PAGE_SIZES[0]) return null
  const from = page.at * page.size + 1
  const to = Math.min(page.total, from + page.size - 1)
  const pages = Math.ceil(page.total / page.size)
  return (
    <nav className={cn('pager')} aria-label="Pages">
      <span className={cn('pager-range')}>{`${noun} ${from} to ${to} of ${page.total.toLocaleString('en-US')}`}</span>
      <label className={cn('pager-size')}>
        Rows per page
        <select value={page.size} onChange={(e) => page.setSize(Number(e.target.value))}>
          {PAGE_SIZES.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>
      <button type="button" disabled={page.at === 0} onClick={() => page.go(page.at - 1)}>
        <ChevronLeft size={14} aria-hidden /> Previous
      </button>
      <button type="button" disabled={page.at >= pages - 1} onClick={() => page.go(page.at + 1)}>
        Next <ChevronRight size={14} aria-hidden />
      </button>
    </nav>
  )
}
