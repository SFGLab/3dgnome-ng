import { Expand, Camera, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { Annotations, Structure } from './api'
import { stretch } from './chain'
import { readPalette, type Mode } from './colors'
import { bp as fmt } from './format'
import { Scene, type Marker, type SceneHandle } from './Scene'
import { saveDataUrl } from './savePng'
import { chainColors, beadAt, chainAt, FOCUS_REACH, focusRanges } from './structure'

export type HoverInfo = { lines: string[]; x: number; y: number; index: number }

type Props = {
  structure: Structure
  annotations: Annotations
  region: [number, number]
  chrom?: string
  index: number
  total: number
  mode: Mode
  onMode: (m: Mode) => void
  markers?: Marker[]
  highlight?: [number, number, number][][]
  /** The beads of a selection, a gene's promoter and enhancer or a measured pair. A short
   * stretch of the chain at each stays solid and the rest is drawn see through. */
  focusBeads?: number[]
  onPickBead?: (bead: number) => void
  /** The genomic position of the chain under the pointer, null when the pointer leaves it. */
  onHoverPosition?: (bp: number | null) => void
  /** Drawn between the toolbar and the 3D box. */
  above?: ReactNode
  /** A stretch of the genome, in bp, whose chain stays solid while the rest is see through. */
  focusSpan?: [number, number] | null
  toolbarStart?: ReactNode
  toolbarEnd?: ReactNode
}

const MODES: [Mode, string][] = [
  ['elements', 'Promoters and enhancers'],
  ['position', 'Genomic position'],
  ['factor', 'Loop factor'],
]
const NO_BEADS: number[] = []

/** The Structure tab's 3D view: toolbar, coloured chain, hover box, legend and caption. */
export function StructureView({ structure, annotations, region, chrom = '', index, total, mode, onMode, markers, highlight, focusBeads = NO_BEADS, onPickBead, onHoverPosition, above, focusSpan = null, toolbarStart, toolbarEnd }: Props) {
  const scene = useRef<SceneHandle>(null)
  const box = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<HoverInfo | null>(null)
  // The focus follows the pointer with a short delay and leaves with a longer one, so a pointer
  // that crosses the model does not flash the whole chain.
  const [held, setHeld] = useState<number | null>(null)
  const over = hover?.index ?? null
  useEffect(() => {
    const t = setTimeout(() => setHeld(over), over === null ? 250 : 120)
    return () => clearTimeout(t)
  }, [over])
  // Keyed on the beads' numbers: a parent passes a new array each render and a new focus would
  // recolour the whole chain on every pointer move.
  const beadsKey = focusBeads.join(',')
  const focus = useMemo(() => {
    const n = structure.chain.length
    const ranges = focusRanges(structure, focusBeads)
    if (held !== null && held < n) ranges.push(stretch(held, n, FOCUS_REACH))
    if (focusSpan) ranges.push([chainAt(structure.chain, focusSpan[0]), chainAt(structure.chain, focusSpan[1])])
    return ranges
  }, [structure, beadsKey, held, focusSpan?.[0], focusSpan?.[1]]) // eslint-disable-line react-hooks/exhaustive-deps
  const palette = useMemo(readPalette, [])
  const points = useMemo(() => structure.chain.map(([x, y, z]) => [x, y, z] as [number, number, number]), [structure])
  // Keyed on the region's numbers: a parent passes a new array each render and a new colour
  // array would rebuild every tube of the chain on every hover.
  const colors = useMemo(
    () => chainColors(structure, annotations, region, mode, palette),
    [structure, annotations, region[0], region[1], mode, palette], // eslint-disable-line react-hooks/exhaustive-deps
  )

  const describe = (i: number): string[] => {
    const k = beadAt(structure.beads, structure.chain[i][3])
    const [s, e] = structure.beads[k]
    const lines = [`${chrom}${chrom ? ':' : ''}${fmt(s)}-${fmt(e)}`]
    const gene = annotations.genes.find((g) => Math.abs(g.tss - (s + e) / 2) <= 5000)
    lines.push(`bead ${fmt(k + 1)}${gene ? `, at the TSS of ${gene.name}` : ''}`)
    const anchor = structure.anchors.find((a) => a.bead === k)
    if (anchor && anchor.loops.length) {
      const factors = [...new Set(anchor.loops.map((l) => l.factor))].join(' and ')
      lines.push(`${factors} anchor, ${anchor.loops.length} loop${anchor.loops.length === 1 ? '' : 's'}`)
    }
    return lines
  }

  const legend: [string, string][] =
    mode === 'elements'
      ? [
          ['promoters', palette.promoter],
          ['enhancers', palette.enhancer],
          ['chromatin', palette.chromatin],
        ]
      : mode === 'factor'
        ? [
            ['CTCF anchors', palette.ctcf],
            ['RNAPII anchors', palette.rnapii],
            ['chromatin', palette.chromatin],
          ]
        : [
            ['region start', '#D6E6F8'],
            ['region end', '#1C3F7A'],
          ]

  return (
    <div className={above ? 'structure-view with-above' : 'structure-view'}>
      <div className="view-toolbar">
        {toolbarStart}
        <div className="segmented" role="group" aria-label="Colour by">
          {MODES.map(([id, label]) => (
            <button key={id} type="button" aria-pressed={mode === id} onClick={() => onMode(id)}>
              {label}
            </button>
          ))}
        </div>
        {toolbarEnd}
        {/* One group, so the three wrap together and none is left alone on a row. */}
        <div className="tool-group">
        <button type="button" className="icon-tool" aria-label="Reset view" onClick={() => scene.current?.reset()}>
          <RotateCcw size={15} aria-hidden />
        </button>
        <button
          type="button"
          className="icon-tool"
          aria-label="Save as PNG"
          onClick={() => {
            const url = scene.current?.png()
            if (url) saveDataUrl(url, `structure_${index}`)
          }}
        >
          <Camera size={15} aria-hidden />
        </button>
        <button type="button" className="icon-tool" aria-label="Full screen" onClick={() => void box.current?.requestFullscreen?.()}>
          <Expand size={15} aria-hidden />
        </button>
        </div>
      </div>
      {above}
      <div className="view-box" ref={box}>
        <Scene
          ref={scene}
          points={points}
          colors={colors}
          markers={markers}
          highlight={highlight}
          focus={focus}
          hovered={hover?.index ?? null}
          onHover={(i, x, y) => {
            setHover(i === null ? null : { lines: describe(i), x, y, index: i })
            onHoverPosition?.(i === null ? null : structure.chain[i][3])
          }}
          onPick={(i) => onPickBead?.(beadAt(structure.beads, structure.chain[i][3]))}
        />
        {hover && (
          <div className="hover-box" style={{ left: hover.x + 22, top: hover.y + 22 }}>
            {hover.lines.map((l, i) => (
              <span key={i} className={i === 0 ? 'mono' : undefined}>
                {l}
              </span>
            ))}
          </div>
        )}
        <ul className="view-legend">
          {legend.map(([label, color]) => (
            <li key={label}>
              <span className="swatch" style={{ background: color }} />
              {label}
            </li>
          ))}
        </ul>
      </div>
      <p className="view-caption">
        Structure {index} of {total}, about one bead per kb. Drag to rotate and scroll to zoom. Hover over a bead to see its position. The chain at the pointer and at each selected bead stays solid and the rest turns see through. Click two beads to measure the distance between them.
      </p>
    </div>
  )
}
