import { useRef, useState } from 'react'
import { ApiError, type Region, type UploadResult, uploadFile, uploadLink } from '../../api'
import { Dialog } from '../../components/Dialog'
import { readHicWindow } from '../hicLocal'
import { guessFactor, NO_OWN, type OwnChoice, type OwnExpression } from '../upload'
import { ContactsStep } from './ContactsStep'
import { type EnhSource, EnhancersStep } from './EnhancersStep'
import { type Entry, fromResult, named, passed, pending } from './entry'
import { ExpressionStep } from './ExpressionStep'
import { LoopsStep } from './LoopsStep'
import { OneFile } from './parts'
import { scoped } from '../../lib/css'
import sheet1 from './owndata.module.css'
import sheet2 from '../model.module.css'

const cn = scoped(sheet1, sheet2)

export type OwnTab = 'loops' | 'anchors' | 'contacts' | 'enhancers' | 'expression'
/** The steps that take one file. */
type Single = 'anchors' | 'contacts' | 'enhancers' | 'expression'

const CONTACT_RES = 25_000
const STEPS: OwnTab[] = ['loops', 'anchors', 'contacts', 'enhancers', 'expression']
const STEP_TITLE: Record<OwnTab, string> = { loops: 'Loops', anchors: 'Anchors', contacts: 'Contacts', enhancers: 'Enhancers', expression: 'Expression' }

/** The dialog for a job's own data: loops per factor, anchors, contacts, enhancers and an
 * expression table, each checked by the server on arrival. Use this data gives the checked uploads to the form. */
export function OwnData({
  cell,
  region,
  initial,
  tab = 'loops',
  maxMb = 20,
  ownCell = false,
  cells = [],
  onClose,
  onUse,
}: {
  cell: string
  region: Region | null
  initial: OwnChoice | null
  tab?: OwnTab
  maxMb?: number
  ownCell?: boolean
  cells?: string[]
  onClose: () => void
  onUse: (own: OwnChoice) => void
}) {
  const start = initial ?? NO_OWN
  const next = useRef(100)
  const [active, setActive] = useState<OwnTab>(tab)
  const [loops, setLoops] = useState<Entry[]>(start.loops.map((l, i) => fromResult(i, l.factor, l.upload)))
  const [single, setSingle] = useState<Record<Single, Entry | null>>({
    anchors: start.anchors && fromResult(90, '', start.anchors),
    contacts: start.contacts && fromResult(91, '', start.contacts),
    enhancers: start.enhancers && fromResult(92, '', start.enhancers),
    expression: start.expression ? fromResult(93, '', start.expression.upload) : null,
  })
  const table = single.expression
  const where = { cell, region: region ? `${region.chrom}:${region.start}-${region.end}` : '' }
  const borrowed = start.enhancersFrom && start.enhancersFrom !== 'none' ? start.enhancersFrom : ''
  const [enhSource, setEnhSource] = useState<EnhSource>(start.enhancers || !ownCell ? 'upload' : borrowed ? 'listed' : 'none')
  const [enhFrom, setEnhFrom] = useState(borrowed || cells[0] || '')
  // The samples the job takes from the expression table and what its values are.
  const [samples, setSamples] = useState<string[]>(start.expression?.columns ?? [])
  const [valueKind, setValueKind] = useState<OwnExpression['kind']>(start.expression?.kind ?? 'counts')


  /** Puts a file into the slot of its kind and follows its upload: the entry is there at once and
   * the outcome is written into it when it arrives, unless another file took the slot meanwhile. */
  const follow = (kind: Single, name: string, size: number, work: () => Promise<UploadResult>, failed: (e: Error) => Partial<Entry> = (e) => ({ failure: e.message })) => {
    const key = next.current++
    setSingle((s) => ({ ...s, [kind]: { key, factor: '', name, size } }))
    const put = (patch: Partial<Entry>) => setSingle((s) => (s[kind]?.key === key ? { ...s, [kind]: { ...s[kind]!, ...patch } } : s))
    return work().then(
      (result) => {
        put({ result })
        return result
      },
      (e: Error) => {
        put(failed(e))
        return undefined
      },
    )
  }
  const remove = (kind: Single) => setSingle((s) => ({ ...s, [kind]: null }))

  const addLoops = (files: FileList | null) => {
    Array.from(files ?? []).forEach((file, i) => {
      const key = next.current++
      const factor = guessFactor(file.name, loops.length + i)
      setLoops((ls) => [...ls, { key, factor, name: file.name, size: file.size }])
      const put = (patch: Partial<Entry>) => setLoops((now) => now.map((l) => (l.key === key ? { ...l, ...patch } : l)))
      uploadFile('loops', file, { ...where, factor }).then(
        (result) => put({ result }),
        (e: Error) => put({ failure: e.message }),
      )
    })
  }
  const addSingle = (kind: Single, file: File) => follow(kind, file.name, file.size, () => uploadFile(kind, file, where))
  const addTable = (file: File) =>
    void addSingle('expression', file).then((result) => {
      if (!result) return
      setSamples(result.ok ? result.samples : [])
      setValueKind(result.counts ? 'counts' : 'values')
    })
  const toggleSample = (sample: string) => setSamples((now) => (table?.result?.samples ?? []).filter((s) => (s === sample ? !now.includes(s) : now.includes(s))))

  const addLink = (url: string) =>
    void follow('contacts', url.split('?')[0].split('/').pop() || url, 0, () => uploadLink(url, where), (e) => ({ failure: e.message, fix: e instanceof ApiError ? (e.fix ?? '') : '' }))
  const addLocal = (file: File) => {
    if (!region) {
      const refusal = { failure: 'The region is not set', fix: 'Enter the region in step 1 first. The page reads only the bins of the region from the file.' }
      return void follow('contacts', file.name, file.size, () => Promise.reject(new Error(refusal.failure)), () => refusal)
    }
    const win = { chrom: region.chrom, start: region.start, end: region.end }
    const read = () => readHicWindow(file, win, CONTACT_RES).then((text) => uploadFile('contacts', new File([text], `${file.name.replace(/\.hic$/i, '')}.${region.chrom}_${region.start}_${region.end}.bedpe`), where))
    void follow('contacts', file.name, file.size, read, (e) => ({ failure: `${file.name} could not be read`, fix: e.message }))
  }

  const kinds = enhSource === 'upload' ? (['anchors', 'contacts', 'enhancers'] as const) : (['anchors', 'contacts'] as const)
  const blocking = [...loops.map((l) => [l, `${l.factor} file`] as const), ...kinds.map((k) => [single[k], `${k} file`] as const), [table, 'expression table'] as const].find(([e]) => !passed(e) && !pending(e))
  const checking = loops.some(pending) || Object.values(single).some(pending)
  const anything = loops.length > 0 || Object.values(single).some(Boolean) || (ownCell && enhSource === 'listed')
  const noSample = !!table?.result?.ok && samples.length === 0
  const use = () =>
    onUse({
      loops: loops.map((l) => ({ factor: l.factor.trim() || 'loops', upload: l.result! })),
      anchors: single.anchors?.result ?? null,
      contacts: single.contacts?.result ?? null,
      enhancers: enhSource === 'upload' ? (single.enhancers?.result ?? null) : null,
      loopsFrom: start.loopsFrom,
      enhancersFrom: ownCell ? (enhSource === 'listed' ? enhFrom : enhSource === 'none' ? 'none' : undefined) : undefined,
      expression: table?.result?.ok ? { upload: table.result, columns: samples, kind: valueKind } : null,
    })

  // The subtitle of each step names what the job will use from it.
  const uses: Record<OwnTab, string> = {
    loops: loops.length ? `${loops.length} file${loops.length === 1 ? '' : 's'}` : 'the listed sets',
    anchors: named(single.anchors) ?? 'the loop ends',
    contacts: named(single.contacts) ?? (ownCell ? 'needed' : "the cell type's Hi-C"),
    enhancers: !ownCell ? (named(single.enhancers) ?? "the cell type's") : enhSource === 'upload' ? (named(single.enhancers) ?? 'none') : enhSource === 'listed' ? enhFrom : 'none',
    expression: named(table) ?? 'none',
  }
  const at = STEPS.indexOf(active)

  return (
    <Dialog id="own-title" name="own-data" title="Your own data" sub="Files stay private to this job and are deleted after 14 days." onClose={onClose}>
      <div className={cn('wizard')}>
        <div className="wizard-steps" role="tablist" aria-orientation="vertical" aria-label="Steps">
          {STEPS.map((id, i) => (
            <button key={id} type="button" role="tab" aria-selected={active === id} onClick={() => setActive(id)}>
              <span className={cn('step-number')} aria-hidden>
                {i + 1}
              </span>
              <span className={cn('step-text')}>
                <span className={cn('step-title')}>{STEP_TITLE[id]}</span>
                <span className={cn('step-uses')}>{uses[id]}</span>
              </span>
            </button>
          ))}
        </div>
        <div className="dialog-body" role="tabpanel">
          {active === 'loops' && <LoopsStep cell={cell} region={region} loops={loops} setLoops={setLoops} addLoops={addLoops} />}
          {active === 'anchors' && (
            <OneFile
              kind="anchors"
              rule="One BED file. The fourth column is optional and gives the motif orientation, L, R or N. Without a file the anchors are the loop ends, merged where they overlap."
              columns="chr start end orientation"
              entry={single.anchors}
              onFile={(f) => void addSingle('anchors', f)}
              onRemove={() => remove('anchors')}
            />
          )}
          {active === 'contacts' && (
            <ContactsStep region={region} maxMb={maxMb} entry={single.contacts} onLink={addLink} onLocal={addLocal} onFile={(f) => void addSingle('contacts', f)} onRemove={() => remove('contacts')} />
          )}
          {active === 'enhancers' && (
            <EnhancersStep
              ownCell={ownCell}
              cells={cells}
              enhSource={enhSource}
              setEnhSource={setEnhSource}
              enhFrom={enhFrom}
              setEnhFrom={setEnhFrom}
              entry={single.enhancers}
              onFile={(f) => void addSingle('enhancers', f)}
              onRemove={() => remove('enhancers')}
            />
          )}
          {active === 'expression' && (
            <ExpressionStep
              table={table}
              samples={samples}
              valueKind={valueKind}
              setValueKind={setValueKind}
              toggleSample={toggleSample}
              onFile={addTable}
              onRemove={() => {
                remove('expression')
                setSamples([])
              }}
            />
          )}
        </div>
      </div>
      <footer className="dialog-foot">
        <p className={cn(`foot-note${blocking || noSample ? ' bad' : ''}`)}>
          {blocking
            ? `Fix or remove the ${blocking[1]} to continue.`
            : noSample
              ? 'Choose one sample or more of the expression table.'
              : checking
                ? 'Checking the files.'
                : active === 'contacts' && !ownCell
                  ? "The map replaces the cell type's map for this job."
                  : ''}
        </p>
        <button type="button" className="secondary" onClick={onClose}>
          Cancel
        </button>
        <button type="button" className="secondary" disabled={at === 0} onClick={() => setActive(STEPS[at - 1])}>
          Back
        </button>
        <button type="button" className="secondary" disabled={at === STEPS.length - 1} onClick={() => setActive(STEPS[at + 1])}>
          Next
        </button>
        <button type="button" className="primary" disabled={!!blocking || checking || noSample || !anything} onClick={use}>
          Use this data
        </button>
      </footer>
    </Dialog>
  )
}
