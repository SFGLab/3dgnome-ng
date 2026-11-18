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

/** Text that can be the start of a gene name: two characters or more that are not coordinates. */
const maybeGene = (text: string) => text.length >= 2 && !text.includes(':')

/** The region field. Typed letters list the genes that match under the field, each with its
 * position. A click, or the arrow keys and Enter, puts a gene's name in the field. */
export function RegionField({ cell, value, onChange }: { cell: string; value: string; onChange: (v: string) => void }) {
  const region = useRegion(cell, value)
  const text = value.trim()
  const typed = useDebounced(text, 150)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const genes = useQuery({ queryKey: ['genes', cell, typed], queryFn: () => api.genes(cell, typed), enabled: open && maybeGene(typed), staleTime: 60_000, retry: false })
  const found = open && maybeGene(text) && typed === text ? (genes.data ?? []) : []
  // A list of the one gene already in the field offers nothing.
  const options = found.length === 1 && found[0].name.toUpperCase() === text.toUpperCase() ? [] : found
  const pick = (name: string) => {
    onChange(name)
    setOpen(false)
  }
  // A part of a gene name is not an error while the list offers its genes.
  const failed = region.isError && options.length === 0
  return (
    <div className="region-field">
      <label htmlFor="region">Region, chr:start-end or a gene name</label>
      <div className={`region-input${failed ? ' error' : ''}`}>
        <input
          id="region"
          role="combobox"
          aria-expanded={options.length > 0}
          aria-controls="region-genes"
          aria-autocomplete="list"
          aria-activedescendant={options[active] ? `region-gene-${active}` : undefined}
          value={value}
          placeholder="TNFRSF8 or chr1:11,500,000-12,600,000"
          autoComplete="off"
          spellCheck={false}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onChange={(e) => {
            onChange(e.target.value)
            setOpen(true)
            setActive(0)
          }}
          onKeyDown={(e) => {
            if (!options.length) return
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
              e.preventDefault()
              setActive((active + (e.key === 'ArrowDown' ? 1 : options.length - 1)) % options.length)
            } else if (e.key === 'Enter') {
              e.preventDefault()
              pick(options[Math.min(active, options.length - 1)].name)
            } else if (e.key === 'Escape') {
              setOpen(false)
            }
          }}
        />
        {region.data && <span className="resolved">{region.data.label}</span>}
      </div>
      {options.length > 0 && (
        <ul id="region-genes" role="listbox" aria-label="Genes" className="gene-options">
          {options.map((g, i) => (
            // The press is taken before the field loses the focus, which would close the list.
            <li key={g.name} id={`region-gene-${i}`} role="option" aria-selected={i === active} className={i === active ? 'active' : undefined} onMouseDown={(e) => { e.preventDefault(); pick(g.name) }} onMouseEnter={() => setActive(i)}>
              <span className="gene-name">{g.name}</span>
              <span className="gene-place">{`${g.chrom}:${g.tss.toLocaleString('en-US')}`}</span>
            </li>
          ))}
        </ul>
      )}
      {failed && <p className="field-error">{region.error.message}</p>}
    </div>
  )
}
