import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { CellTypes } from '../model/CellTypes'
import { EXAMPLES, type Example } from '../model/examples'
import { EMPTY_FORM, type FormState } from '../model/form'
import { LoopPreview } from '../model/LoopPreview'
import { RegionField, useRegion } from '../model/RegionField'
import { SidePanel } from '../model/SidePanel'
import { Step } from '../model/Step'
import '../styles/doc.css'
import './model.css'

function fromExample(e: Example): FormState {
  return { ...EMPTY_FORM, cell: e.cell, region: e.region, factors: e.factors, svSet: e.svSet ?? '', sample: e.sample ?? '' }
}

/** The form that starts a job: cell type and region, loops, variants, background and settings. */
export function Model() {
  const [params] = useSearchParams()
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: api.catalog })
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const set = (patch: Partial<FormState>) => setForm((f) => ({ ...f, ...patch }))
  const region = useRegion(form.cell, form.region)

  useEffect(() => {
    const e = EXAMPLES.find((x) => x.id === params.get('example'))
    if (e) setForm(fromExample(e))
  }, [params])

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
          {catalog.data && <CellTypes catalog={catalog.data} value={form.cell} onChange={(cell) => set({ cell })} />}
          <RegionField cell={form.cell} value={form.region} onChange={(region) => set({ region })} />
          {region.data && <LoopPreview cell={form.cell} region={region.data} factors={form.factors} />}
        </Step>
        <Step n={2} title="Loops">
          <div />
        </Step>
        <Step n={3} title="Structural variants" note="optional">
          <div />
        </Step>
        <Step n={4} title="Contact background">
          <div />
        </Step>
        <Step n={5} title="Ensemble and simulation settings">
          <div />
        </Step>
      </div>
    </div>
  )
}
