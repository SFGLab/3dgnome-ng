import { Download } from 'lucide-react'
import { type Cell, saveCsv } from '../job/saveCsv'

/** A button that saves every row of a table as a CSV file, also the rows of its other pages. */
export function CsvButton({ file, head, rows }: { file: string; head: string[]; rows: () => Cell[][] }) {
  return (
    <button type="button" className="chip-button csv-button" aria-label={`Save the table as ${file}.csv`} onClick={() => saveCsv(file, head, rows())}>
      <Download size={13} aria-hidden /> <span className="mono">CSV</span>
    </button>
  )
}
