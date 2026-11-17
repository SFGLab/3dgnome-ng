import { useQuery } from '@tanstack/react-query'
import type { JobApi, Summary } from './api'
import { GenesTab } from './GenesTab'
import { parseRegion } from './StructureTab'

/** The Genes tab with its rows from the job's gene endpoint. */
export function GenesPanel({ api, summary, filter, selected, onSelect }: { api: JobApi; summary: Summary; filter: string; selected: string | null; onSelect: (g: string) => void }) {
  const genes = useQuery({ queryKey: ['genes', api.base], queryFn: api.genes })
  if (!genes.data) return <div aria-busy="true" />
  return (
    <GenesTab
      rows={genes.data}
      chrom={parseRegion(summary.region)?.chrom ?? ''}
      structures={summary.structures}
      epUrl={api.fileUrl('ep.tsv')}
      filter={filter}
      selected={selected}
      onSelect={onSelect}
    />
  )
}
