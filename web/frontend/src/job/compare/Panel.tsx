import { Unanswered } from '../../components/Unanswered'
import { askedOf, type JobApi } from '../../api'
import { useQuery } from '@tanstack/react-query'
import { Camera, ChevronDown } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { readPalette } from '../colors'
import { FACTORS } from '../../lib/format'
import { type CameraLink } from '../cameraLink'
import { Scene, type SceneHandle } from '../Scene'
import { saveDataUrl } from '../export/savePng'
import { chainColors, focusRanges } from '../structure'
import { parseRegion } from '../../lib/region'
import { parseJobUrl } from '../../routes'
import { scoped } from '../../lib/css'
import sheet from '../job.module.css'

const cn = scoped(sheet)

const factorColor = (f: string) => (f === 'CTCF' ? 'var(--ctcf)' : 'var(--rnapii)')

type Choice = { url: string; name: string }

/** The head of one side. With `onPick` it is a button that opens, under itself, the jobs that
 * side can show and a field for a job link. Without it the head is plain. */
export function PanelHead({ line, title, sub, choices = [], onPick, startOpen = false }: { line?: string; title: string; sub: string; choices?: Choice[]; onPick?: (url: string) => void; startOpen?: boolean }) {
  const [open, setOpen] = useState(startOpen)
  const [pasted, setPasted] = useState('')
  const box = useRef<HTMLDivElement>(null)
  // A press outside the head and its list closes the list.
  useEffect(() => {
    if (!open) return
    const away = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', away)
    return () => document.removeEventListener('pointerdown', away)
  }, [open])
  const pick = (url: string) => {
    setOpen(false)
    setPasted('')
    onPick?.(url)
  }
  const text = (
    <>
      {line && <span className={cn('factor-line')} style={{ background: line }} />}
      <span className={cn('panel-text')}>
        <span className={cn('panel-title')}>{title}</span>
        <span className={cn('panel-sub')}>{sub}</span>
      </span>
    </>
  )
  if (!onPick) return <div className={cn('panel-head fixed')}>{text}</div>
  return (
    <div className={cn('panel-pick')} ref={box} onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}>
      <button type="button" className={cn('panel-head')} aria-haspopup="true" aria-expanded={open} onClick={() => setOpen(!open)}>
        {text}
        <ChevronDown size={14} aria-hidden />
      </button>
      {open && (
        <div className={cn('panel-menu')} role="group" aria-label="Jobs to compare with">
          {choices.map((j) => (
            <button key={j.url} type="button" className={cn('menu-item')} onClick={() => pick(j.url)}>
              {j.name}
            </button>
          ))}
          <div className={cn('paste')}>
            <input placeholder="Paste a job link" value={pasted} onChange={(e) => setPasted(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && parseJobUrl(pasted) && pick(pasted)} />
            <button type="button" className="action" disabled={!parseJobUrl(pasted)} onClick={() => pick(pasted)}>
              Compare
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export function Panel({
  api,
  label,
  structure: index,
  gene,
  link,
  linkId,
  pair,
  choices,
  onPick,
}: {
  api: JobApi
  label: string
  structure: number
  gene: string | null
  link?: CameraLink
  linkId: string
  /**
   * The enhancer of the selected gene in the comparison for this job and its distance. Null for a gene the
   * comparison does not hold, which is then not marked.
   */
  pair?: { bead?: number; mean: number; sd: number } | null
  choices?: Choice[]
  onPick?: (url: string) => void
}) {
  const summary = useQuery(askedOf(api).summary())
  const structure = useQuery(askedOf(api).structure(index))
  const annotations = useQuery(askedOf(api).annotations())
  const genes = useQuery(askedOf(api).genes())
  const palette = useMemo(readPalette, [])
  const scene = useRef<SceneHandle>(null)
  const s = summary.data
  const region = parseRegion(s?.region ?? null)
  const colors = useMemo(
    () => (structure.data && annotations.data && region ? chainColors(structure.data, annotations.data, [region.start, region.end], 'elements', palette) : []),
    [structure.data, annotations.data, region?.start, region?.end, palette], // eslint-disable-line react-hooks/exhaustive-deps
  )
  // The scene builds its tubes anew for a new list of points. The list is kept for a structure.
  const points = useMemo(() => (structure.data?.chain ?? []).map(([x, y, z]) => [x, y, z] as [number, number, number]), [structure.data])
  const row = pair === null ? undefined : genes.data?.find((r) => r.gene === gene)
  const enhancerBead = pair?.bead ?? row?.enhancer_bead
  // The promoter and the enhancer of the selected gene, marked A and B as on the Structure tab.
  const markers =
    row && structure.data
      ? [row.promoter_bead, enhancerBead!]
          .filter((k) => structure.data!.beads[k])
          .map((k, i) => ({ label: 'AB'[i], at: [structure.data!.beads[k][2], structure.data!.beads[k][3], structure.data!.beads[k][4]] as [number, number, number] }))
      : []
  // The chain at the promoter and the enhancer of the selected gene is
  // solid and the rest see through.
  const focus = useMemo(
    () => (row && structure.data ? focusRanges(structure.data, [row.promoter_bead, enhancerBead!]) : undefined),
    [structure.data, row?.promoter_bead, enhancerBead], // eslint-disable-line react-hooks/exhaustive-deps
  )
  return (
    <div className={cn('compare-panel')}>
      <PanelHead
        line={factorColor(s?.request.factors === 'CTCF' ? 'CTCF' : 'RNAPII')}
        title={s ? `${FACTORS[s.request.factors] ?? s.request.factors} loops` : ' '}
        sub={s ? `${label}, ${s.request.cell_type}, ${s.structures} structures` : ''}
        choices={choices}
        onPick={onPick}
      />
      <div className={cn('view-box compare-box')}>
        {(summary.isError || structure.isError) && <Unanswered of={[summary, structure]} />}
        {structure.data && (
          <Scene ref={scene} points={points} colors={colors} markers={markers} focus={focus} link={link} linkId={linkId} />
        )}
        {structure.data && (
          <button
            type="button"
            className={cn('icon-tool panel-save')}
            aria-label={`Save the structure of ${label} as a PNG`}
            onClick={() => {
              const url = scene.current?.png(row ? `A to B ${(pair?.mean ?? row.distance).toFixed(1)} ± ${(pair?.sd ?? row.sd).toFixed(1)}` : undefined)
              if (url) saveDataUrl(url, `structure_${linkId === 'a' ? 'this_job' : 'other_job'}_${index}`)
            }}
          >
            <Camera size={15} aria-hidden />
          </button>
        )}
        {row && (
          <span className={cn('ab-label')}>
            A to B <strong>{`${(pair?.mean ?? row.distance).toFixed(1)} ± ${(pair?.sd ?? row.sd).toFixed(1)}`}</strong>
          </span>
        )}
      </div>
    </div>
  )
}
