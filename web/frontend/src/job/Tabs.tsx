import { Box, Download, FileTerminal, Grid3x3, Link2, List, Spline, SquareSplitHorizontal } from 'lucide-react'
import type { ReactNode } from 'react'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

export type Tab = 'structure' | 'genes' | 'contacts' | 'loops' | 'log'

const TABS: [Tab, string, ReactNode][] = [
  ['structure', 'Structure', <Box key="i" size={15} aria-hidden />],
  ['genes', 'Genes', <List key="i" size={15} aria-hidden />],
  ['contacts', 'Contact map', <Grid3x3 key="i" size={15} aria-hidden />],
  ['loops', 'Loops', <Spline key="i" size={15} aria-hidden />],
  ['log', 'Log', <FileTerminal key="i" size={15} aria-hidden />],
]

type Props = {
  tab: Tab
  counts: Partial<Record<Tab, number>>
  comparing: boolean
  zipUrl: string
  onTab: (t: Tab) => void
  onCompare: () => void
}

/** The tabs of the explorer and its actions: compare, copy the link, download every file. */
export function JobTabs({ tab, counts, comparing, zipUrl, onTab, onCompare }: Props) {
  return (
    <div className={cn('job-tabs')}>
      <div role="tablist" aria-label="Job views">
        {TABS.map(([id, label, icon]) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id} className="job-tab" onClick={() => onTab(id)}>
            {icon}
            {label}
            {counts[id] !== undefined && <span className={cn('count')}>{counts[id]}</span>}
          </button>
        ))}
      </div>
      <div className={cn('job-actions')}>
        <button type="button" className={cn(comparing ? 'action dark' : 'action')} onClick={onCompare}>
          <SquareSplitHorizontal size={14} aria-hidden /> {comparing ? 'Comparing' : 'Compare with another job'}
        </button>
        <button type="button" className="action" onClick={() => void navigator.clipboard?.writeText(window.location.href)}>
          <Link2 size={14} aria-hidden /> Copy link
        </button>
        <a className={cn('action dark')} href={zipUrl}>
          <Download size={14} aria-hidden /> Download all
        </a>
      </div>
    </div>
  )
}
