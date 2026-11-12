import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import type { JobApi, Summary } from './api'
import type { Mode } from './colors'
import { StructureView } from './StructureView'

export function parseRegion(region: string | null): { chrom: string; start: number; end: number } | null {
  const m = region?.match(/^(\w+):(\d+)-(\d+)$/)
  return m ? { chrom: m[1], start: Number(m[2]), end: Number(m[3]) } : null
}

/** The Structure tab: the selected structure of the ensemble in 3D. */
export function StructureTab({ api, summary, selected }: { api: JobApi; summary: Summary; selected: number }) {
  const [mode, setMode] = useState<Mode>('elements')
  const structure = useQuery({ queryKey: ['structure', api.base, selected], queryFn: () => api.structure(selected) })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const region = parseRegion(summary.region)
  if (!structure.data || !annotations.data || !region) return <div className="view-box" aria-busy="true" />
  return (
    <StructureView
      structure={structure.data}
      annotations={annotations.data}
      region={[region.start, region.end]}
      chrom={region.chrom}
      index={selected}
      total={summary.structures}
      mode={mode}
      onMode={setMode}
    />
  )
}
