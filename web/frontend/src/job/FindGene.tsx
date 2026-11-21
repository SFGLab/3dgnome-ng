import { Search } from 'lucide-react'
import { useState } from 'react'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

/** The Find a gene field: Enter selects the gene of that name, case insensitive. */
export function FindGene({ names, onFind }: { names: string[]; onFind: (gene: string) => void }) {
  const [text, setText] = useState('')
  const [miss, setMiss] = useState(false)
  const find = () => {
    const hit = names.find((n) => n.toUpperCase() === text.trim().toUpperCase())
    setMiss(!hit)
    if (hit) onFind(hit)
  }
  return (
    <label className={cn(`find-gene${miss ? ' miss' : ''}`)}>
      <Search size={14} aria-hidden />
      <input placeholder="Find a gene" value={text} list="gene-names" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && find()} />
      <datalist id="gene-names">
        {names.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
    </label>
  )
}
