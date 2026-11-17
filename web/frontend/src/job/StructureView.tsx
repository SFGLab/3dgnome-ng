import { Expand, Camera, RotateCcw } from 'lucide-react'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import type { Annotations, Structure } from './api'
import { colorAt, readPalette, type Mode } from './colors'
import { bp as fmt } from './format'
import { Scene, type Marker, type SceneHandle } from './Scene'

export type HoverInfo = { lines: string[]; x: number; y: number; index: number }

/** The colour of every chain point of a structure under a colour mode. */
export function chainColors(structure: Structure, annotations: Annotations, region: [number, number], mode: Mode, palette = readPalette()): string[] {
  const ctx = {
    palette,
    start: region[0],
    end: region[1],
    promoters: annotations.genes.map((g) => [g.tss - 1000, g.tss + 1000] as [number, number]),
    enhancers: annotations.enhancers,
    anchors: structure.anchors.map((a) => ({ start: a.start, end: a.end, factors: a.loops.map((l) => l.factor) })),
  }
  return structure.chain.map((p) => colorAt(mode, p[3], ctx))
}

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
  onPickBead?: (bead: number) => void
  toolbarStart?: ReactNode
  toolbarEnd?: ReactNode
}

const MODES: [Mode, string][] = [
  ['elements', 'Promoters and enhancers'],
  ['position', 'Genomic position'],
  ['factor', 'Loop factor'],
]

/** Index of the bead whose span holds a genomic position, by binary search over starts. */
export function beadAt(beads: Structure['beads'], bp: number): number {
  let lo = 0
  let hi = beads.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (beads[mid][0] <= bp) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** The Structure tab's 3D view: toolbar, coloured chain, hover box, legend and caption. */
export function StructureView({ structure, annotations, region, chrom = '', index, total, mode, onMode, markers, highlight, onPickBead, toolbarStart, toolbarEnd }: Props) {
  const scene = useRef<SceneHandle>(null)
  const box = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<HoverInfo | null>(null)
  const palette = useMemo(readPalette, [])
  const points = useMemo(() => structure.chain.map(([x, y, z]) => [x, y, z] as [number, number, number]), [structure])
  // Keyed on the region's numbers: a parent passes a new array each render, and a new colour
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
    <div className="structure-view">
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
        <button type="button" className="icon-tool" aria-label="Reset view" onClick={() => scene.current?.reset()}>
          <RotateCcw size={15} aria-hidden />
        </button>
        <button
          type="button"
          className="icon-tool"
          aria-label="Save as PNG"
          onClick={() => {
            const url = scene.current?.png()
            if (!url) return
            const a = document.createElement('a')
            a.href = url
            a.download = `structure_${index}.png`
            a.click()
          }}
        >
          <Camera size={15} aria-hidden />
        </button>
        <button type="button" className="icon-tool" aria-label="Full screen" onClick={() => void box.current?.requestFullscreen?.()}>
          <Expand size={15} aria-hidden />
        </button>
      </div>
      <div className="view-box" ref={box}>
        <Scene
          ref={scene}
          points={points}
          colors={colors}
          markers={markers}
          highlight={highlight}
          hovered={hover?.index ?? null}
          onHover={(i, x, y) => setHover(i === null ? null : { lines: describe(i), x, y, index: i })}
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
        Structure {index} of {total}, about one bead per kb. Drag to rotate and scroll to zoom. Hover over a bead to see its position. Click two beads to measure the distance between them.
      </p>
    </div>
  )
}
