import { useEffect, useState } from 'react'

/** Rows a page of a table holds. */
export const PAGE = 50

export type Page<T> = { rows: T[]; at: number; total: number; size: number; go: (at: number) => void }

/** One page of rows. `reset` changes when a filter or sort does, and sends the table back to
 * its first page. */
export function usePage<T>(all: T[], reset: unknown = null, size = PAGE): Page<T> {
  const [at, setAt] = useState(0)
  useEffect(() => setAt(0), [reset])
  const last = Math.max(0, Math.ceil(all.length / size) - 1)
  const here = Math.min(at, last)
  return { rows: all.slice(here * size, here * size + size), at: here, total: all.length, size, go: (n) => setAt(Math.max(0, Math.min(n, last))) }
}
