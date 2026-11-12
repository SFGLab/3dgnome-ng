import { Upload, X } from 'lucide-react'
import type { Catalog } from '../api'
import type { FormState } from './form'

type Props = {
  catalog: Catalog
  svSet: string
  sample: string
  haplotype: 0 | 1
  vcf: File | null
  onChange: (patch: Partial<FormState>) => void
  onOpen: (mode: 'hosted' | 'upload', file: File | null) => void
}

/** Step 3, a hosted call set with its sample and haplotype, or an uploaded VCF. A VCF and the
 * variants of a chosen sample open the dialog that shows their change to the loops. */
export function Variants({ catalog, svSet, sample, haplotype, vcf, onChange, onOpen }: Props) {
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
          <select value={sample} disabled={!svSet} onChange={(e) => onChange({ sample: e.target.value })}>
            <option value="">e.g. HG00733</option>
            {samples.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
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
          <button type="button" className="icon-button" aria-label="Remove the VCF" onClick={() => onChange({ vcf: null })}>
            <X size={14} aria-hidden />
          </button>
        </div>
      ) : (
        <label className="upload">
          <Upload size={14} aria-hidden />
          <span>Or drop your own VCF here. The file must be compressed with bgzip and have SVTYPE of DEL, INS or INV.</span>
          <input type="file" accept=".gz,.bgz" onChange={(e) => e.target.files?.[0] && onOpen('upload', e.target.files[0])} />
        </label>
      )}
      {((svSet && sample) || vcf) && (
        <button type="button" className="link-button" onClick={() => onOpen(vcf ? 'upload' : 'hosted', vcf)}>
          Show the variants in the region
        </button>
      )}
    </div>
  )
}
