import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useDebounced, useRegion } from './useRegion'

/** Text that can be the start of a gene name: two characters or more that are not coordinates. */
const maybeGene = (text: string) => text.length >= 2 && !text.includes(':')

/**
 * The region field. Typed letters list the matching genes under the field. The form gets
 * coordinates as they are typed and a gene name once it is chosen, Enter is pressed or the field is
 * left. With `quiet` a refused region marks the field and its reason is left to the caller, which
 * shows it elsewhere.
 */
export function RegionField({ cell, value, quiet = false, onChange }: { cell: string; value: string; quiet?: boolean; onChange: (v: string) => void }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const region = useRegion(cell, value)
  const text = draft.trim()
  const typed = useDebounced(text, 150)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  // The list scrolls, and the arrow keys bring the gene they reach into view.
  const list = useRef<HTMLUListElement>(null)
  useEffect(() => {
    list.current?.children[active]?.scrollIntoView?.({ block: 'nearest' })
  }, [active])
  const settled = draft === value
  const genes = useQuery({ queryKey: ['genes', cell, typed], queryFn: () => api.genes(cell, typed), enabled: open && maybeGene(typed), staleTime: 60_000, retry: false })
  const found = open && maybeGene(text) && typed === text ? (genes.data ?? []) : []
  // A list of the one gene already in the field offers nothing.
  const options = found.length === 1 && found[0].name.toUpperCase() === text.toUpperCase() ? [] : found
  const commit = (v: string) => {
    setDraft(v)
    if (v !== value) onChange(v)
  }
  const pick = (name: string) => {
    commit(name)
    setOpen(false)
  }
  // A name still being typed is not an error, nor is a part of a name the list offers genes for.
  const failed = settled && region.isError && options.length === 0
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
          value={draft}
          placeholder="TNFRSF8 or chr1:11,500,000-12,600,000"
          autoComplete="off"
          spellCheck={false}
          onFocus={() => setOpen(true)}
          onBlur={() => {
            setOpen(false)
            commit(draft)
          }}
          onChange={(e) => {
            const v = e.target.value
            setDraft(v)
            // Coordinates, and an emptied field, reach the form at once.
            if (v.includes(':') || v.trim() === '') onChange(v)
            setOpen(true)
            setActive(0)
          }}
          onKeyDown={(e) => {
            if (!options.length) {
              if (e.key === 'Enter') commit(draft)
              return
            }
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
        {settled && region.data && <span className="resolved">{region.data.label}</span>}
      </div>
      {options.length > 0 && (
        <ul id="region-genes" ref={list} role="listbox" aria-label="Genes" className="gene-options">
          {options.map((g, i) => (
            // The press is taken before the field loses the focus, which would close the list.
            <li key={g.name} id={`region-gene-${i}`} role="option" aria-selected={i === active} className={i === active ? 'active' : undefined} onMouseDown={(e) => { e.preventDefault(); pick(g.name) }} onMouseEnter={() => setActive(i)}>
              <span className="gene-name">{g.name}</span>
              <span className="gene-place">{`${g.chrom}:${g.tss.toLocaleString('en-US')}`}</span>
            </li>
          ))}
        </ul>
      )}
      {failed && !quiet && <p className="field-error">{region.error.message}</p>}
    </div>
  )
}
