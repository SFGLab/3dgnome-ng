/** The region the form has, resolved by the server a moment after the last key. */
import { asked } from '../api'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

/** Resolves the text of the region field, a gene name or coordinates, through the API as the user types. */
export function useRegion(cell: string, text: string) {
  const q = useDebounced(text.trim(), 250)
  return useQuery({ ...asked.region(cell, q), enabled: q.length > 1, retry: false })
}
