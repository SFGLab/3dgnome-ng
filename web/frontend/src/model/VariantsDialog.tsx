import { asked, type Catalog, type Region, type VariantRow, type UploadResult, uploadFile } from '../api'
import { useQueries, useQuery } from '@tanstack/react-query'
import { CircleAlert, CircleCheck, FileArchive, LoaderCircle, Upload, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Pager } from '../components/Pager'
import { usePage } from '../components/usePage'
import type { FormState } from './form'
import { type OwnChoice, ownField } from './upload'
import { megabytes } from '../lib/format'
import '../styles/dialog.css'
import './own/owndata.css'
import './variants.css'

export type VariantsMode = 'hosted' | 'upload'

const NAMES: Record<string, string> = {
  DEL: 'Deletion',
  INS: 'Insertion',
  INV: 'Inversion',
  DUP: 'Duplication',
  BND: 'Breakend',
  CNV: 'Copy number change',
}

/** A length in bp, kb or Mb. */
function basePairs(n: number): string {
  if (n < 1000) return `${n} bp`
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)} kb`
  return `${(n / 1_000_000).toFixed(1)} Mb`
}

const W = 780
const H = 110
const AXIS = 62

/** The loops of the region above the axis, own loops in place of the cell type's and the
 * variants below it. */
function Track({ cell, region, factors, own, rows }: { cell: string; region: Region; factors: string; own: OwnChoice | null; rows: VariantRow[] }) {
  const span = `${region.chrom}:${region.start}-${region.end}`
  const mine = own?.loops ?? []
  const preview = useQuery({ ...asked.preview(cell, span), enabled: !mine.length })
  const ownArcs = useQueries({
    queries: mine.map((l) => asked.uploadPreview(l.upload.id, cell, span)),
  })
  const x = (p: number) => Math.min(W, Math.max(0, ((p - region.start) / (region.end - region.start)) * W))
  const shown = factors === 'CTCF+RNAPII' ? ['CTCF', 'RNAPII'] : [factors]
  return (
    <svg className="sv-track" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Variants in ${span}`}>
      {ownArcs.flatMap((q, k) =>
        (q.data ?? []).map(([a, b], i) => {
          const r = Math.max((x(b) - x(a)) / 2, 1)
          return <path key={`own${k}-${i}`} className={`arc own f${Math.min(k, 2)}`} d={`M ${x(a)} ${AXIS} A ${r} ${Math.min(r, AXIS - 8)} 0 0 1 ${x(b)} ${AXIS}`} />
        }),
      )}
      {(mine.length ? [] : shown).flatMap((f) =>
        (preview.data?.loops[f] ?? []).map(([a, b], i) => {
          const r = Math.max((x(b) - x(a)) / 2, 1)
          return <path key={`${f}${i}`} className={`arc ${f.toLowerCase()}`} d={`M ${x(a)} ${AXIS} A ${r} ${Math.min(r, AXIS - 8)} 0 0 1 ${x(b)} ${AXIS}`} />
        }),
      )}
      <line className="axis" x1={0} x2={W} y1={AXIS} y2={AXIS} />
      {rows.map((v, i) => {
        const x1 = x(v.pos)
        const w = Math.max(x(v.end) - x1, 4)
        return (
          <g key={i} className={`sv-mark sv-${v.type.toLowerCase()}`}>
            <rect x={x1} y={AXIS + 6} width={w} height={10} rx={1} />
            <text x={x1 + w / 2} y={AXIS + 30} textAnchor="middle">
              {v.type}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

/** The structural variants dialog: a hosted call set or an uploaded VCF, its sample and
 * haplotype and the change each variant makes to the loops of the region. */
export function VariantsDialog(props: {
  catalog: Catalog
  cell: string
  region: Region | null
  factors: string
  mode: VariantsMode
  file: File | null
  upload?: UploadResult | null
  svSet: string
  sample: string
  haplotype: 0 | 1
  own?: OwnChoice | null
  onClose: () => void
  onApply: (patch: Partial<FormState>) => void
}) {
  const { catalog, cell, region, factors, onClose, onApply } = props
  const [mode, setMode] = useState<VariantsMode>(props.mode)
  const [file, setFile] = useState<File | null>(props.file)
  const [upload, setUpload] = useState<UploadResult | null>(props.upload ?? null)
  const [failure, setFailure] = useState('')
  const [svSet, setSvSet] = useState(props.svSet || Object.keys(catalog.sv_sets)[0] || '')
  // One sample for each source, so a look at the other source does not lose the choice.
  const [chosen, setChosen] = useState<Record<VariantsMode, string>>({ hosted: props.mode === 'hosted' ? props.sample : '', upload: props.mode === 'upload' ? props.sample : '' })
  const sample = chosen[mode]
  const setSampleOf = (of: VariantsMode, value: string | ((s: string) => string)) => setChosen((all) => ({ ...all, [of]: typeof value === 'function' ? value(all[of]) : value }))
  const setSample = (value: string) => setSampleOf(mode, value)
  const [haplotype, setHaplotype] = useState<0 | 1>(props.haplotype)
  const span = region ? `${region.chrom}:${region.start}-${region.end}` : ''

  useEffect(() => {
    if (!file) return
    setUpload(null)
    setFailure('')
    let live = true
    uploadFile('vcf', file, { cell, region: span })
      .then((u) => {
        if (!live) return
        setUpload(u)
        setSampleOf('upload', (s) => (u.samples.includes(s) ? s : (u.samples[0] ?? '')))
      })
      .catch((e: Error) => live && setFailure(e.message))
    return () => {
      live = false
    }
    // The upload belongs to the file alone; the region only words its check.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  const source = mode === 'upload' ? (upload?.ok ? upload.id : '') : svSet
  const own = props.own?.loops.length ? props.own : null
  const fields: Record<string, string> = {
    cell,
    region: span,
    factors,
    sample,
    haplotype: String(haplotype),
    ...(mode === 'upload' ? { upload: source } : { sv_set: source }),
    ...(own ? { own: ownField({ ...own, contacts: null, enhancers: null }) } : {}),
  }
  const check = useQuery({
    ...asked.variants(fields),
    enabled: !!span && !!sample && !!source,
    retry: false,
  })

  // The table goes back to its first page when the check is for other fields. The fields are a new
  // object at every render and their text is what changes with them.
  const variantPage = usePage(check.data?.variants ?? [], JSON.stringify(fields))
  const samples = mode === 'upload' ? (upload?.samples ?? []) : (catalog.sv_sets[svSet]?.samples ?? [])
  const n = check.data?.variants.length
  const ready = !!sample && !!source
  const apply = () => onApply(mode === 'upload' ? { vcfUpload: upload, svSet: '', sample, haplotype } : { svSet, sample, haplotype, vcfUpload: null })

  const haplotypeField = (
    <label className="field">
      <span>Haplotype</span>
      <select value={haplotype} onChange={(e) => setHaplotype(e.target.value === '1' ? 1 : 0)}>
        <option value="0">first</option>
        <option value="1">second</option>
      </select>
    </label>
  )
  const sampleField = (label: string) => (
    <label className="field">
      <span>{label}</span>
      <select value={sample} disabled={!samples.length} onChange={(e) => setSample(e.target.value)}>
        {!samples.includes(sample) && <option value="">{samples.length ? 'choose a sample' : 'no samples yet'}</option>}
        {samples.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
    </label>
  )

  return (
    <div className="dialog-backdrop">
      <div className="dialog sv-dialog" role="dialog" aria-modal="true" aria-labelledby="sv-title">
        <header className="dialog-head">
          <div>
            <h2 id="sv-title">Structural variants</h2>
            <p className="dialog-sub">The variants change the loops before modelling. An uploaded VCF stays private to this job.</p>
          </div>
          <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </header>
        <div className="dialog-body sv-body">
          <div className="segmented" role="tablist">
            {(
              [
                ['hosted', 'Hosted call set'],
                ['upload', 'Upload a VCF'],
              ] as const
            ).map(([id, title]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={mode === id}
                disabled={id === 'hosted' && !Object.keys(catalog.sv_sets).length}
                onClick={() => setMode(id)}
              >
                {title}
              </button>
            ))}
          </div>
          <div className="sv-fields" role="tabpanel">
            {mode === 'upload' ? (
              file || upload ? (
                <span className="file-chip">
                  <FileArchive size={14} aria-hidden />
                  <span className="file-name">{file?.name ?? upload?.name}</span>
                  <span className="file-size">{megabytes(file?.size ?? upload?.size ?? 0)}</span>
                </span>
              ) : (
                <label className="add-file">
                  <Upload size={14} aria-hidden /> Choose a VCF compressed with bgzip
                  <input type="file" accept=".gz,.bgz" className="visually-hidden" onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])} />
                </label>
              )
            ) : (
              <label className="field">
                <span>Call set</span>
                <select
                  value={svSet}
                  onChange={(e) => {
                    setSvSet(e.target.value)
                    setSample('')
                  }}
                >
                  {Object.entries(catalog.sv_sets).map(([name, s]) => (
                    <option key={name} value={name}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {sampleField(mode === 'upload' ? 'Sample, from the VCF header' : 'Sample')}
            {haplotypeField}
          </div>
          {mode === 'upload' && file && !upload && !failure && (
            <p className="check">
              <LoaderCircle size={14} aria-hidden className="spin" /> Reading the samples of the VCF.
            </p>
          )}
          {(failure || upload?.ok === false) && (
            <p className="sv-box bad" role="alert">
              <CircleAlert size={15} aria-hidden /> {failure || `${upload?.error?.found} ${upload?.error?.fix}`}
            </p>
          )}
          {!span && <p className="hint">Enter the region in step 1 to see the variants inside it.</p>}
          {check.isError && (
            <p className="sv-box bad" role="alert">
              <CircleAlert size={15} aria-hidden /> {check.error.message}
            </p>
          )}
          {check.isFetching && (
            <p className="check">
              <LoaderCircle size={14} aria-hidden className="spin" /> Applying the variants to the loops of the region.
            </p>
          )}
          {check.data && region && (
            <>
              <p className="sv-box good">
                <CircleCheck size={15} aria-hidden /> {check.data.summary}
              </p>
              <Track cell={cell} region={region} factors={factors} own={own} rows={check.data.variants} />
              {check.data.variants.length > 0 && (
                <>
                  <Pager page={variantPage} noun="variants" />
                  <div className="sv-table-wrap">
                    <table className="sv-table">
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>Position</th>
                          <th>Length</th>
                          <th>Genotype</th>
                          <th>Change to the loops</th>
                        </tr>
                      </thead>
                      <tbody>
                        {variantPage.rows.map((v, i) => (
                          <tr key={i}>
                            <td>
                              <span className={`sv-dot sv-${v.type.toLowerCase()}`} aria-hidden />
                              {NAMES[v.type] ?? v.type}
                            </td>
                            <td className="mono">{`${region.chrom}:${v.pos.toLocaleString('en-US')}`}</td>
                            <td>{basePairs(v.length)}</td>
                            <td className="mono">{v.genotype}</td>
                            <td>{v.change}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </>
          )}
        </div>
        <footer className="dialog-foot">
          <p className="foot-note">The haplotype of a phased genotype decides which variants apply. An unphased variant goes to the first haplotype.</p>
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="primary" disabled={!ready} onClick={apply}>
            {n === undefined ? 'Apply the variants' : `Apply ${n} variant${n === 1 ? '' : 's'}`}
          </button>
        </footer>
      </div>
    </div>
  )
}
