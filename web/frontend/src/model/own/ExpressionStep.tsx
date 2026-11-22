import type { OwnExpression } from '../upload'
import type { Entry } from './entry'
import { OneFile } from './parts'
import { scoped } from '../../lib/css'
import sheet from './owndata.module.css'

const cn = scoped(sheet)

/** The expression step, with a table of genes by samples, the samples the job takes and the kind of its values. */
export function ExpressionStep({
  table,
  samples,
  valueKind,
  setValueKind,
  toggleSample,
  onFile,
  onRemove,
}: {
  table: Entry | null
  samples: string[]
  valueKind: OwnExpression['kind']
  setValueKind: (kind: OwnExpression['kind']) => void
  toggleSample: (sample: string) => void
  onFile: (f: File) => void
  onRemove: () => void
}) {
  return (
    <OneFile
      kind="expression"
      rule="One table of genes by samples, separated by tabs or commas, for the Genes tab and the comparison of two jobs. The first column holds the gene as an Ensembl id or a gene name. Every other column of numbers is a sample. A column named length is optional and gives the gene length in bp. The output of featureCounts is read as it is. Without a table the job has the expression chosen in the form."
      columns="gene length sample_1 sample_2"
      entry={table}
      onFile={onFile}
      onRemove={onRemove}
    >
      {table?.result?.ok && (
        <>
          <fieldset className={cn('own-samples')}>
            <legend>Samples of this job</legend>
            {table.result.samples.map((s) => (
              <label key={s} className={cn('own-sample')}>
                <input type="checkbox" checked={samples.includes(s)} onChange={() => toggleSample(s)} />
                {s}
              </label>
            ))}
          </fieldset>
          <label className={cn('own-select')}>
            <span>The values are</span>
            <select value={valueKind} onChange={(e) => setValueKind(e.target.value === 'values' ? 'values' : 'counts')}>
              <option value="counts">read counts</option>
              <option value="values">normalised, such as TPM or TMM</option>
            </select>
          </label>
          <p className={cn('hint')}>
            {valueKind === 'counts'
              ? 'Two jobs with two samples or more each are compared with DESeq2. With one sample on a side the comparison is a ratio with no test.'
              : 'Two jobs with normalised values are compared by the ratio of their values, with no test.'}
          </p>
        </>
      )}
    </OneFile>
  )
}
