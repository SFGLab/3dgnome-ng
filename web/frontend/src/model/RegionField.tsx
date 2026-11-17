import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api, ApiError, type Region } from '../api'

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

/** The region field, resolving a gene name or coordinates through the API as the user types. */
export function useRegion(cell: string, text: string) {
  const q = useDebounced(text.trim(), 250)
  return useQuery<Region, ApiError>({ queryKey: ['region', cell, q], queryFn: () => api.region(cell, q), enabled: q.length > 1, retry: false })
}

export function RegionField({ cell, value, onChange }: { cell: string; value: string; onChange: (v: string) => void }) {
  const region = useRegion(cell, value)
  return (
    <div className="region-field">
      <label htmlFor="region">Region, chr:start-end or a gene name</label>
      <div className={`region-input${region.isError ? ' error' : ''}`}>
        <input id="region" value={value} placeholder="TNFRSF8 or chr1:11,500,000-12,600,000" onChange={(e) => onChange(e.target.value)} autoComplete="off" />
        {region.data && <span className="resolved">{region.data.label}</span>}
      </div>
      {region.isError && <p className="field-error">{region.error.message}</p>}
    </div>
  )
}
