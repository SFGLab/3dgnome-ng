/** Saving the rows of a table as a CSV file. */

export type Cell = string | number | null | undefined

/** A cell as CSV writes it: in quotes when it holds a comma, a quote or a line break. */
const cell = (v: Cell) => {
  const text = v === null || v === undefined ? '' : String(v)
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** The text of a CSV file with a head line and one line for each row. */
export function csvText(head: string[], rows: Cell[][]): string {
  return [head, ...rows].map((r) => r.map(cell).join(',')).join('\r\n') + '\r\n'
}

/** Saves a table as `<file>.csv`. */
export function saveCsv(file: string, head: string[], rows: Cell[][]): void {
  const link = document.createElement('a')
  link.href = URL.createObjectURL(new Blob([csvText(head, rows)], { type: 'text/csv' }))
  link.download = `${file}.csv`
  link.click()
  URL.revokeObjectURL(link.href)
}
