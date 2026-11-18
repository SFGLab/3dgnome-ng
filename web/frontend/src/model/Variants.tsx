import { useEffect, useState } from 'react'
import { ChevronRight, Dna, Upload, X } from 'lucide-react'
import type { Catalog } from '../api'
import type { FormState } from './form'
import type { UploadResult } from './upload'

type Props = {
  catalog: Catalog
  svSet: string
  sample: string
  haplotype: 0 | 1
  vcfUpload: UploadResult | null
  onChange: (patch: Partial<FormState>) => void
  onOpen: (mode: 'hosted' | 'upload', file: File | null) => void
}

/** A call set with more samples than this gets a field to type in. */
const LONG_LIST = 60

/** The sample of a large call set, typed with the browser's own suggestions. The form takes the
 * sample once the text names one of the set's samples, and has none while it does not. */
function SampleSearch({ samples, sample, onChange }: { samples: string[]; sample: string; onChange: (sample: string) => void }) {
  const [text, setText] = useState(sample)
  useEffect(() => setText(sample), [sample])
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
          const v = e.target.value.trim()
          if (known.has(v)) onChange(v)
          else if (sample) onChange('')
        }}
      />
      <datalist id="sample-names">
        {samples.map((s) => (
          <option key={s} value={s} />
        ))}
      </datalist>
    </>
  )
}

/** Step 3, a hosted call set with its sample and haplotype, or an uploaded VCF. A VCF and the
 * variants of a chosen sample open the dialog that shows their change to the loops. */
export function Variants({ catalog, svSet, sample, haplotype, vcfUpload: vcf, onChange, onOpen }: Props) {
  const samples = svSet ? (catalog.sv_sets[svSet]?.samples ?? []) : []
  return (
    <div className="variants">
      <div className="fields">
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
            <SampleSearch key={svSet} samples={samples} sample={sample} onChange={(v) => onChange({ sample: v })} />
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
      {vcf ? (
        <div className="upload chosen">
          <Upload size={14} aria-hidden />
          <span>{vcf.name}</span>
          <button type="button" className="icon-button" aria-label="Remove the VCF" onClick={() => onChange({ vcfUpload: null })}>
            <X size={14} aria-hidden />
          </button>
        </div>
      ) : (
        <label className="upload">
          <Upload size={14} aria-hidden />
          <span>Or drop your own VCF here. The file must be compressed with bgzip and have SVTYPE of DEL, INS, INV or DUP.</span>
          <input type="file" accept=".gz,.bgz" onChange={(e) => e.target.files?.[0] && onOpen('upload', e.target.files[0])} />
        </label>
      )}
      {((svSet && sample) || vcf) && (
        <button type="button" className="preview-button" onClick={() => onOpen(vcf ? 'upload' : 'hosted', null)}>
          <Dna size={15} aria-hidden />
          <span>Show the variants in the region</span>
          <ChevronRight size={14} className="chevron" aria-hidden />
        </button>
      )}
    </div>
  )
}
