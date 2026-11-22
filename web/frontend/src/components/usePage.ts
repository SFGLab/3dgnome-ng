import { useEffect, useState } from 'react'

/** Rows a page of a table holds until the reader chooses another count. */
export const PAGE = 50
/** The counts the reader can choose. */
export const PAGE_SIZES = [25, 50, 100, 250]
const KEY = 'gnome3d.rows'

export type Page<T> = { rows: T[]; at: number; total: number; size: number; go: (at: number) => void; setSize: (size: number) => void }

/** Returns the last choice of rows per page of the reader, kept in the browser. */
function remembered(fallback: number): number {
  try {
    const n = Number(localStorage.getItem(KEY))
    return PAGE_SIZES.includes(n) ? n : fallback
  } catch {
    return fallback
  }
}

/**
 * Returns one page of rows. `reset` changes with a filter or a sort and sends the table back to its
 * first page, as a change of the rows per page does.
 */
export function usePage<T>(all: T[], reset: unknown = null, size = PAGE): Page<T> {
  const [at, setAt] = useState(0)
  const [rows, setRows] = useState(() => remembered(size))
  useEffect(() => setAt(0), [reset])
  const last = Math.max(0, Math.ceil(all.length / rows) - 1)
  const here = Math.min(at, last)
  return {
    rows: all.slice(here * rows, here * rows + rows),
    at: here,
    total: all.length,
    size: rows,
    go: (n) => setAt(Math.max(0, Math.min(n, last))),
    setSize: (n) => {
      setRows(n)
      setAt(0)
      try {
        localStorage.setItem(KEY, String(n))
      } catch {
        // A browser without storage keeps the choice for this page view alone.
      }
    },
  }
}
