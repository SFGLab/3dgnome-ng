import { useEffect, useState } from 'react'
import { ChevronRight, Dna, Upload, X } from 'lucide-react'
import type { Catalog, SampleRow, UploadResult } from '../api'
import type { FormState } from './form'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

type Props = {
  catalog: Catalog
  svSet: string
  sample: string
  haplotype: 0 | 1
  vcfUpload: UploadResult | null
  /** Sex, population and parents of the call set's samples, for a call set that has them. */
  sampleTable?: SampleRow[]
  /** False until step 1 has a region. The variants are shown for a region and the button waits for one. */
  hasRegion?: boolean
  onChange: (patch: Partial<FormState>) => void
  onOpen: (mode: 'hosted' | 'upload', file: File | null) => void
}

/** A call set with more samples than this gets a field to type in. */
const LONG_LIST = 60

/** Returns the population, the superpopulation and the sex of a sample, the known ones. */
const describe = (r: SampleRow) => [r.population, r.superpopulation, r.sex].filter(Boolean).join(', ')

/** A population of this many samples or fewer is listed whole. */
const LISTED = 12

/**
 * The sample of a large call set, typed with the suggestions of the browser. The form takes the
 * sample once the text is the name of a sample of the set and has none while it is not.
 */
function SampleSearch({ samples, sample, table, onChange, onText }: { samples: string[]; sample: string; table: Map<string, SampleRow>; onChange: (sample: string) => void; onText: (text: string) => void }) {
  const [text, setText] = useState(sample)
  useEffect(() => {
    setText(sample)
    if (sample) onText(sample)
  }, [sample]) // eslint-disable-line react-hooks/exhaustive-deps
  const known = new Set(samples)
  return (
    <>
      <input
        type="text"
        list="sample-names"
        value={text}
        placeholder={`type one of ${samples.length.toLocaleString('en-US')}, e.g. ${samples[0]}`}
        autoComplete="off"
        spellCheck={false}
        aria-invalid={text !== '' && !known.has(text.trim())}
        onChange={(e) => {
          setText(e.target.value)
          onText(e.target.value)
          const v = e.target.value.trim()
          if (known.has(v)) onChange(v)
          else if (sample) onChange('')
        }}
      />
      <datalist id="sample-names">
        {samples.map((s) => {
          const row = table.get(s)
          // The description is the option's label attribute. As text it would join the name of the field.
          return <option key={s} value={s} label={row ? describe(row) : undefined} />
        })}
      </datalist>
    </>
  )
}

/** The chosen sample's population, sex and family, each relative a button that chooses it. With
 * no sample chosen and the code of a population typed, the samples of that population. */
function SampleFacts({ rows, table, sample, text, onPick }: { rows: SampleRow[]; table: Map<string, SampleRow>; sample: string; text: string; onPick: (sample: string) => void }) {
  const pick = (name: string) => (
    <button key={name} type="button" className="link-button" onClick={() => onPick(name)}>
      {name}
    </button>
  )
  const list = (names: string[]) => names.flatMap((n, i) => (i ? [', ', pick(n)] : [pick(n)]))
  const row = table.get(sample)
  if (row) {
    const children = rows.filter((r) => r.father === sample || r.mother === sample).map((r) => r.sample)
    return (
      <p className={cn('sample-facts')}>
        {`${sample}: ${describe(row) || 'no record'}.`}
        {row.father && <> Father {pick(row.father)}.</>}
        {row.mother && <> Mother {pick(row.mother)}.</>}
        {children.length > 0 && (
          <>
            {children.length === 1 ? ' Child ' : ' Children '}
            {list(children)}.
          </>
        )}
      </p>
    )
  }
  const code = text.trim().toUpperCase()
  const members = code ? rows.filter((r) => r.population === code || r.superpopulation === code).map((r) => r.sample) : []
  if (!members.length) return null
  const shown = members.length <= LISTED ? members : members.slice(0, LISTED - 2)
  return (
    <p className={cn('sample-facts')}>
      {`${code}, ${members.length.toLocaleString('en-US')} samples: `}
      {list(shown)}
      {shown.length < members.length ? ` and ${(members.length - shown.length).toLocaleString('en-US')} more.` : '.'}
    </p>
  )
}

/** Step 3, a hosted call set with its sample and haplotype, or an uploaded VCF. A VCF and the
 * variants of a chosen sample open the dialog that shows their change to the loops. */
export function Variants({ catalog, svSet, sample, haplotype, vcfUpload: vcf, sampleTable, hasRegion = true, onChange, onOpen }: Props) {
  const samples = svSet ? (catalog.sv_sets[svSet]?.samples ?? []) : []
  const [typed, setTyped] = useState('')
  const rows = sampleTable ?? []
  const table = new Map(rows.map((r) => [r.sample, r]))
  return (
    <div className={cn('variants')}>
      <div className={cn('fields')}>
        <label className="field">
          <span>Hosted call set</span>
          <select value={svSet} disabled={vcf !== null} onChange={(e) => onChange({ svSet: e.target.value, sample: '' })}>
            <option value="">none</option>
            {Object.entries(catalog.sv_sets).map(([name, s]) => (
              <option key={name} value={name}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Sample</span>
          {samples.length > LONG_LIST ? (
            <SampleSearch key={svSet} samples={samples} sample={sample} table={table} onChange={(v) => onChange({ sample: v })} onText={setTyped} />
          ) : (
            <select value={sample} disabled={!svSet} onChange={(e) => onChange({ sample: e.target.value })}>
              <option value="">e.g. HG00733</option>
              {samples.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          )}
        </label>
        <label className="field">
          <span>Haplotype</span>
          <select value={haplotype} onChange={(e) => onChange({ haplotype: e.target.value === '1' ? 1 : 0 })}>
            <option value="0">first</option>
            <option value="1">second</option>
          </select>
        </label>
      </div>
      {!vcf && rows.length > 0 && <SampleFacts rows={rows} table={table} sample={sample} text={typed} onPick={(v) => onChange({ sample: v })} />}
      {vcf ? (
        <div className={cn('upload chosen')}>
          <Upload size={14} aria-hidden />
          <span>{vcf.name}</span>
          <button type="button" className="icon-button" aria-label="Remove the VCF" onClick={() => onChange({ vcfUpload: null })}>
            <X size={14} aria-hidden />
          </button>
        </div>
      ) : (
        <label className={cn('upload')}>
          <Upload size={14} aria-hidden />
          <span>Or drop your own VCF here. The file must be compressed with bgzip and have SVTYPE of DEL, INS, INV or DUP.</span>
          <input type="file" accept=".gz,.bgz" onChange={(e) => e.target.files?.[0] && onOpen('upload', e.target.files[0])} />
        </label>
      )}
      {((svSet && sample) || vcf) && !hasRegion && <p className={cn('sample-facts')}>Enter the region in step 1 to see the variants inside it.</p>}
      {((svSet && sample) || vcf) && hasRegion && (
        <button type="button" className={cn('preview-button')} onClick={() => onOpen(vcf ? 'upload' : 'hosted', null)}>
          <Dna size={15} aria-hidden />
          <span>Show the variants in the region</span>
          <ChevronRight size={14} className={cn('chevron')} aria-hidden />
        </button>
      )}
    </div>
  )
}
