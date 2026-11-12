import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowRight } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, ApiError } from '../api'
import { Background } from '../model/Background'
import { CellTypes } from '../model/CellTypes'
import { EXAMPLES, type Example } from '../model/examples'
import { EMPTY_FORM, type FormState } from '../model/form'
import { LoopOptions } from '../model/LoopOptions'
import { LoopPreview } from '../model/LoopPreview'
import { OwnData, type OwnTab } from '../model/OwnData'
import { rememberJob } from '../model/recent'
import { RegionField, useRegion } from '../model/RegionField'
import { changedSettings } from '../model/settings'
import { SidePanel } from '../model/SidePanel'
import { EnsembleFields, SettingsTree } from '../model/SettingsTree'
import { Step } from '../model/Step'
import { ownField } from '../model/upload'
import { Variants } from '../model/Variants'
import '../styles/doc.css'
import './model.css'

const FIELD_STEP: Record<string, number> = {
  cell_type: 1,
  region: 1,
  own: 1,
  contacts: 1,
  factors: 2,
  loops: 2,
  sv_set: 3,
  sample: 3,
  background: 4,
  n_models: 5,
  overrides: 5,
}

/** The form as the multipart body of POST /api/jobs. */
function toFormData(form: FormState, overrides: Record<string, string>): FormData {
  const d = new FormData()
  d.append('cell_type', form.cell)
  d.append('region', form.region)
  d.append('factors', form.factors)
  d.append('background', form.background)
  d.append('sv_set', form.svSet)
  d.append('sample', form.sample)
  d.append('haplotype', String(form.haplotype))
  d.append('n_models', String(form.structures))
  d.append('overrides', JSON.stringify(overrides))
  if (form.vcf) d.append('vcf', form.vcf)
  const own = form.own
  if (own.loops.length || own.anchors || own.contacts || own.enhancers) d.append('own', ownField(own))
  return d
}

function goTo(url: string) {
  window.location.assign(url)
}

function fromExample(e: Example): FormState {
  return { ...EMPTY_FORM, cell: e.cell, region: e.region, factors: e.factors, svSet: e.svSet ?? '', sample: e.sample ?? '' }
}

/** The form that starts a job: cell type and region, loops, variants, background and settings. */
export function Model({ onSubmitted = goTo }: { onSubmitted?: (url: string) => void }) {
  const [params] = useSearchParams()
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: api.catalog })
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const set = (patch: Partial<FormState>) => setForm((f) => ({ ...f, ...patch }))
  const [dialog, setDialog] = useState<{ tab: OwnTab; cell: boolean } | null>(null)
  const ownCell = form.cell === 'own'
  // Your own cell type takes its gene names and server settings from the first listed cell type.
  const lookup = ownCell ? (Object.keys(catalog.data?.cell_types ?? {})[0] ?? 'GM12878') : form.cell
  const region = useRegion(lookup, form.region)
  const schema = useQuery({ queryKey: ['settings', lookup], queryFn: () => api.settings(lookup) })

  useEffect(() => {
    const e = EXAMPLES.find((x) => x.id === params.get('example'))
    if (e) setForm(fromExample(e))
  }, [params])

  const submit = useMutation({
    mutationFn: () => api.submit(toFormData(form, schema.data ? changedSettings(schema.data, form.settings) : {})),
    onSuccess: ({ url }) => {
      const label = region.data ? `${form.cell} ${region.data.label.split(',')[0]}` : `${form.cell} ${form.region}`
      rememberJob({ url, label, name: form.name.trim(), at: Date.now() })
      onSubmitted(url)
    },
  })
  const error = submit.error instanceof ApiError ? submit.error : null
  const errorStep = error ? (FIELD_STEP[error.field ?? ''] ?? 5) : 0
  const stepError = (n: number) =>
    errorStep === n && error ? (
      <p className="field-error" role="alert">
        {error.message}
      </p>
    ) : null

  const cellType = catalog.data?.cell_types[form.cell]
  const hicdiffusionOk = !!catalog.data?.hicdiffusion && !(region.data && region.data.end - region.data.start > 2_000_000)
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
            Pick a cell type and a region of up to 20 Mb, then choose the loops. A job returns an ensemble of 3D structures with promoters and enhancers marked.
          </p>
        </header>
        <Step n={1} title="Cell type and region">
          {catalog.data && (
            <CellTypes catalog={catalog.data} value={form.cell} onChange={(cell) => set({ cell })} onOwn={() => setDialog({ tab: 'loops', cell: true })} />
          )}
          <RegionField cell={lookup} value={form.region} onChange={(region) => set({ region })} />
          {region.data && !ownCell && !form.own.loops.length && <LoopPreview cell={form.cell} region={region.data} factors={form.factors} />}
          {stepError(1)}
        </Step>
        <Step n={2} title="Loops">
          <LoopOptions
            value={form.own.loops.length ? 'own' : form.factors}
            ownCell={ownCell}
            onChange={(factors) => set({ factors, own: { ...form.own, loops: [] } })}
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
          {catalog.data && <Variants catalog={catalog.data} svSet={form.svSet} sample={form.sample} haplotype={form.haplotype} vcf={form.vcf} onChange={set} />}
          {stepError(3)}
        </Step>
        <Step n={4} title="Contact background">
          <Background
            value={form.background}
            region={region.data}
            available={!!catalog.data?.hicdiffusion}
            lymphoblastoid={cellType?.lymphoblastoid ?? true}
            onChange={(background) => set({ background })}
          />
          {stepError(4)}
        </Step>
        <Step n={5} title="Ensemble and simulation settings">
          <EnsembleFields structures={form.structures} name={form.name} max={catalog.data?.max_structures ?? 100} onChange={set} />
          {schema.data && <SettingsTree schema={schema.data} values={form.settings} onChange={(settings) => set({ settings })} />}
          {stepError(5)}
        </Step>
        {dialog && (
          <OwnData
            cell={lookup}
            region={region.data ?? null}
            initial={form.own}
            tab={dialog.tab}
            onClose={() => setDialog(null)}
            onUse={(own) => {
              set(dialog.cell ? { own, cell: 'own' } : { own })
              setDialog(null)
            }}
          />
        )}
        <div className="submit-row">
          <button type="button" className="submit" disabled={!form.region.trim() || submit.isPending} onClick={() => submit.mutate()}>
            {`Build ${form.structures} structure${form.structures === 1 ? '' : 's'}`} <ArrowRight size={16} aria-hidden />
          </button>
          <span className="timing">The job page shows the progress of the job and its place in the queue.</span>
        </div>
      </div>
    </div>
  )
}
