import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowRight, Info } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, ApiError, type Region, jobApi } from '../api'
import { served } from '../base'
import { Background } from '../model/Background'
import { CellTypes } from '../model/CellTypes'
import { EXAMPLES } from '../model/examples'
import { Expression } from '../model/Expression'
import { EMPTY_FORM, longRegion, type FormState, withOwn } from '../model/form'
import { FIELD_STEP, fromExample, fromRequest, readUploads, toFormData } from '../model/request'
import { LoopOptions } from '../model/LoopOptions'
import { LoopPreview } from '../model/LoopPreview'
import { OwnData, type OwnTab } from '../model/own/OwnData'
import { recentJobs, rememberJob } from '../myjobs/recent'
import { RegionField } from '../model/RegionField'
import { useRegion } from '../model/useRegion'
import { changedSettings } from '../model/settings'
import { SidePanel } from '../model/SidePanel'
import { EnsembleFields, SettingsTree } from '../model/SettingsTree'
import { Step } from '../model/Step'
import { Variants } from '../model/Variants'
import { VariantsDialog, type VariantsMode } from '../model/VariantsDialog'
import '../styles/doc.css'
import './model.css'

function goTo(url: string) {
  window.location.assign(served(url))
}

/** The form that starts a job: cell type and region, loops, variants, background and settings. */
export function Model({ onSubmitted = goTo }: { onSubmitted?: (url: string) => void }) {
  const [params] = useSearchParams()
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: api.catalog })
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const set = (patch: Partial<FormState>) => setForm((f) => ({ ...f, ...patch }))
  const [dialog, setDialog] = useState<{ tab: OwnTab; cell: boolean } | null>(null)
  const [svDialog, setSvDialog] = useState<{
    mode: VariantsMode
    file: File | null
  } | null>(null)
  const ownCell = form.cell === 'own'
  // Your own cell type takes its gene names and server settings from the first listed cell type.
  const cells = Object.keys(catalog.data?.cell_types ?? {})
  const lookup = ownCell ? form.own.loopsFrom || cells[0] || 'GM12878' : form.cell
  const region = useRegion(lookup, form.region)
  // The loops stay drawn at the last region while new coordinates are read, so a drag of the
  // region on them does not blank the picture. A region longer than the limit stays on them too,
  // to be shortened there. Any other region the server refuses takes them away.
  const lastRegion = useRef<Region>()
  if (region.data) lastRegion.current = region.data
  const maxSpan = (catalog.data?.max_region_mb ?? 20) * 1e6
  const tooLong = region.isError ? longRegion(form.region, maxSpan) : undefined
  const shownRegion = region.data ?? tooLong ?? (region.isError || !form.region.trim() ? undefined : lastRegion.current)
  const hasSampleTable = Boolean(catalog.data?.sv_sets[form.svSet]?.sample_table)
  const sampleTable = useQuery({ queryKey: ['sample-table', form.svSet], queryFn: () => api.sampleTable(form.svSet), enabled: hasSampleTable, staleTime: Infinity })
  const schema = useQuery({
    queryKey: ['settings', lookup],
    queryFn: () => api.settings(lookup),
  })

  useEffect(() => {
    const e = EXAMPLES.find((x) => x.id === params.get('example'))
    if (e) setForm(fromExample(e))
  }, [params])

  // A link from a job's page fills the form with the request of that job.
  const source = /^\/(jobs|p)\/([A-Za-z0-9_-]{8,64})$/.exec(params.get('from') ?? '')
  const sourceJob = useQuery({
    queryKey: ['request-of', source?.[0]],
    queryFn: () => jobApi(source![1] as 'jobs' | 'p', source![2]).summary(),
    enabled: !!source,
    retry: false,
    staleTime: Infinity,
  })
  // The job as My jobs shows it in this browser, by its name or by its cell type and region.
  const sourceKnown = source ? recentJobs().find((j) => j.url === source[0]) : undefined
  const sourceName = sourceKnown?.name || sourceKnown?.label || ''
  const cellNames = cells.join(',')
  // The form is filled once for a link. A later read of the catalog must not undo what was typed since.
  const filledFrom = useRef('')
  useEffect(() => {
    if (!source || !sourceJob.data || !catalog.data || filledFrom.current === source[0]) return
    filledFrom.current = source[0]
    setForm(fromRequest(sourceJob.data.request, cellNames.split(',').filter(Boolean)))
  }, [source, sourceJob.data, catalog.data, cellNames])

  // My jobs shows a job by its name, or by its cell type and region when it has none. Two jobs
  // shown alike cannot be told apart there.
  const label = region.data ? `${form.cell} ${region.data.label.split(',')[0]}` : `${form.cell} ${form.region}`
  const shownAs = form.name.trim() || label
  const sameName = form.region.trim() !== '' && recentJobs().some((j) => (j.name || j.label) === shownAs)
  const submit = useMutation({
    mutationFn: () => api.submit(toFormData(form, schema.data ? changedSettings(schema.data, form.settings) : {})),
    onSuccess: ({ url }) => {
      rememberJob({ url, label, name: form.name.trim(), at: Date.now() })
      onSubmitted(url)
    },
  })
  // A refusal of the server is shown at the step of its field. A request without an answer is shown under the button.
  const error = submit.error instanceof ApiError && submit.error.status > 0 ? submit.error : null
  const errorStep = error ? (FIELD_STEP[error.field ?? ''] ?? 6) : 0
  const stepError = (n: number) =>
    errorStep === n && error ? (
      <p className="field-error" role="alert">
        {error.message}
      </p>
    ) : null

  const cellType = catalog.data?.cell_types[form.cell]
  // The hosted expression of the cell type, and of the sample of a hosted call set.
  const expressionOptions = useQuery({
    queryKey: ['expression', form.cell, form.svSet, form.sample],
    queryFn: () => api.expression(form.cell, form.svSet, form.sample),
    enabled: !ownCell && !!catalog.data?.cell_types[form.cell],
    staleTime: 60_000,
  })
  // A source the cell type or the sample no longer has is taken off the form.
  const offered = expressionOptions.data
  useEffect(() => {
    if (form.expression && form.expression !== 'own' && offered && !offered.some((o) => o.id === form.expression)) setForm((f) => ({ ...f, expression: '' }))
  }, [form.expression, offered])
  const hicdiffusionMb = catalog.data?.hicdiffusion_mb ?? 2
  const hicdiffusionOk = !!catalog.data?.hicdiffusion && !(region.data && region.data.end - region.data.start > hicdiffusionMb * 1e6)
  useEffect(() => {
    if (form.background === 'hicdiffusion' && catalog.data && !hicdiffusionOk) setForm((f) => ({ ...f, background: 'hic' }))
  }, [form.background, hicdiffusionOk, catalog.data])

  return (
    <div className="doc-page model-page">
      <SidePanel onExample={(e) => setForm(fromExample(e))} />
      <div className="reading model-form">
        <header className="page-head">
          <h1>Model a region</h1>
          <p className="lead">
            {`Pick a cell type and a region of up to ${catalog.data?.max_region_mb ?? 20} Mb, then choose the loops.`} A job returns an ensemble of 3D structures with promoters and enhancers marked.
          </p>
        </header>
        {source && sourceJob.data && (
          <p className="filled-from" role="status">
            The form is filled with the request of <a href={served(source[0])}>{sourceName || `job ${source[2].slice(0, 8)}`}</a>. Change what you need and start a new job.
            {readUploads(sourceJob.data.request) && ' That job read uploaded files. They are not filled in. Add them again in their steps.'}
          </p>
        )}
        {source && sourceJob.isError && (
          <p className="filled-from" role="alert">
            The job of this link was not found and the form is not filled.
          </p>
        )}
        <Step n={1} title="Cell type and region">
          {catalog.data && (
            <CellTypes
              catalog={catalog.data}
              value={form.cell}
              own={form.own}
              onChange={(cell) => set({ cell, own: { ...form.own, loopsFrom: undefined, enhancersFrom: undefined } })}
              onOwn={() => setDialog({ tab: 'loops', cell: true })}
            />
          )}
          <RegionField cell={lookup} value={form.region} quiet={!!tooLong && !form.own.loops.length} onChange={(region) => set({ region })} />
          {shownRegion && !form.own.loops.length && <LoopPreview cell={lookup} region={shownRegion} factors={form.factors} maxSpan={maxSpan} onRegion={(region) => set({ region })} />}
          {stepError(1)}
        </Step>
        <Step n={2} title="Loops">
          <LoopOptions
            value={form.own.loops.length ? 'own' : form.factors}
            ownCell={ownCell}
            cells={cells}
            loopsFrom={lookup}
            onLoopsFrom={(loopsFrom) => set({ own: { ...form.own, loopsFrom } })}
            onChange={(factors) => set({ factors, own: { ...form.own, loops: [], loopsFrom: ownCell ? lookup : undefined } })}
            onOwn={() => setDialog({ tab: 'loops', cell: false })}
          />
          {form.own.loops.length > 0 && (
            <p className="own-summary">
              {`Your own loops: ${form.own.loops.map((l) => `${l.factor} from ${l.upload.name}`).join(', ')}.`}{' '}
              <button type="button" className="link-button" onClick={() => setDialog({ tab: 'loops', cell: ownCell })}>
                Change the files
              </button>
            </p>
          )}
          {stepError(2)}
        </Step>
        <Step n={3} title="Structural variants" note="optional">
          {catalog.data && (
            <Variants
              catalog={catalog.data}
              svSet={form.svSet}
              sample={form.sample}
              haplotype={form.haplotype}
              vcfUpload={form.vcfUpload}
              sampleTable={hasSampleTable ? sampleTable.data : undefined}
              hasRegion={!!region.data}
              onChange={set}
              onOpen={(mode, file) => setSvDialog({ mode, file })}
            />
          )}
          {stepError(3)}
        </Step>
        <Step n={4} title="Contact background">
          <Background
            value={form.background}
            region={region.data}
            available={!!catalog.data?.hicdiffusion}
            limitMb={hicdiffusionMb}
            lymphoblastoid={cellType?.lymphoblastoid ?? true}
            onChange={(background) => set({ background })}
          />
          {stepError(4)}
        </Step>
        <Step n={5} title="Gene expression" note="optional">
          <Expression options={expressionOptions.data ?? []} value={form.expression} cell={form.cell} onChange={(expression) => set({ expression })} own={form.own.expression ?? null} onOpen={() => setDialog({ tab: 'expression', cell: ownCell })} onRemove={() => set({ expression: '', own: { ...form.own, expression: null } })} />
          {stepError(5)}
        </Step>
        <Step n={6} title="Ensemble and simulation settings">
          <EnsembleFields structures={form.structures} name={form.name} max={catalog.data?.max_structures ?? 100} onChange={set} />
          {sameName && (
            <p className="notice" role="status">
              <Info size={14} aria-hidden /> {`My jobs already lists a job as ${shownAs}. Give this job a name of its own to tell the two apart.`}
            </p>
          )}
          {schema.data && <SettingsTree schema={schema.data} values={form.settings} onChange={(settings) => set({ settings })} />}
          {stepError(6)}
        </Step>
        <div className="submit-row">
          <button type="button" className="submit" disabled={!form.region.trim() || submit.isPending} onClick={() => submit.mutate()}>
            {`Build ${form.structures} structure${form.structures === 1 ? '' : 's'}`} <ArrowRight size={16} aria-hidden />
          </button>
          <span className="timing">The job page shows the progress of the job and its place in the queue.</span>
        </div>
        {submit.error && !error && (
          <p className="field-error" role="alert">
            The server did not answer. Check the connection and build again.
          </p>
        )}
      </div>
      {svDialog && catalog.data && (
        <VariantsDialog
          catalog={catalog.data}
          cell={lookup}
          region={region.data ?? null}
          factors={form.factors}
          mode={svDialog.mode}
          file={svDialog.file}
          upload={form.vcfUpload}
          svSet={form.svSet}
          sample={form.sample}
          haplotype={form.haplotype}
          own={form.own}
          onClose={() => setSvDialog(null)}
          onApply={(patch) => {
            set(patch)
            setSvDialog(null)
          }}
        />
      )}
      {dialog && (
        <OwnData
          cell={lookup}
          region={region.data ?? null}
          initial={form.own}
          tab={dialog.tab}
          maxMb={catalog.data?.max_region_mb ?? 20}
          ownCell={dialog.cell || ownCell}
          cells={cells}
          onClose={() => setDialog(null)}
          onUse={(own) => {
            set(withOwn(form, own, dialog, cells))
            setDialog(null)
          }}
        />
      )}
    </div>
  )
}
