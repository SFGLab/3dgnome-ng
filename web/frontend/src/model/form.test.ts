import { describe, expect, it } from 'vitest'
import type { UploadResult } from '../api'
import { EMPTY_FORM, withOwn } from './form'
import { NO_OWN, type OwnChoice } from './upload'

const file = (kind: UploadResult['kind']): UploadResult => ({ id: `${kind}upload1`, kind, name: `${kind}.txt`, size: 1, ok: true, summary: '', error: null, samples: [], counts: false })
const table = { upload: file('expression'), columns: ['a1'], kind: 'counts' as const }
const CELLS = ['GM12878', 'H1ESC']

describe('the form after the own data dialog is confirmed', () => {
  it('keeps the cell type and takes the files for a dialog opened from the loops', () => {
    const own: OwnChoice = { ...NO_OWN, loops: [{ factor: 'CTCF', upload: file('loops') }] }
    expect(withOwn(EMPTY_FORM, own, { cell: false, tab: 'loops' }, CELLS)).toEqual({ own, expression: '' })
  })

  it('makes your own cell type with the loops of the first listed cell type and no enhancers', () => {
    const own: OwnChoice = { ...NO_OWN, contacts: file('contacts') }
    expect(withOwn(EMPTY_FORM, own, { cell: true, tab: 'contacts' }, CELLS)).toEqual({ cell: 'own', expression: '', own: { ...own, loopsFrom: 'GM12878', enhancersFrom: 'none' } })
  })

  it('takes no listed loops where your own cell type has loops, and keeps a chosen listed set', () => {
    const loops = [{ factor: 'CTCF', upload: file('loops') }]
    const withLoops = withOwn(EMPTY_FORM, { ...NO_OWN, loops, contacts: file('contacts'), loopsFrom: 'H1ESC' }, { cell: true, tab: 'loops' }, CELLS)
    expect(withLoops.own?.loopsFrom).toBeUndefined()
    const listed = withOwn({ ...EMPTY_FORM, cell: 'own' }, { ...NO_OWN, contacts: file('contacts'), loopsFrom: 'H1ESC', enhancersFrom: 'H1ESC' }, { cell: false, tab: 'contacts' }, CELLS)
    expect(listed).toEqual({ expression: '', own: { ...NO_OWN, contacts: file('contacts'), loopsFrom: 'H1ESC', enhancersFrom: 'H1ESC' } })
  })

  it('makes a table added in the dialog the expression of the job', () => {
    expect(withOwn(EMPTY_FORM, { ...NO_OWN, expression: table }, { cell: false, tab: 'expression' }, CELLS).expression).toBe('own')
    // A form that had a hosted source keeps it when the dialog was opened for other files and the table was there before.
    const had = { ...EMPTY_FORM, expression: 'ENCODE', own: { ...NO_OWN, expression: table } }
    expect(withOwn(had, { ...NO_OWN, expression: table, contacts: file('contacts') }, { cell: false, tab: 'contacts' }, CELLS).expression).toBe('ENCODE')
    expect(withOwn(had, { ...NO_OWN, expression: table }, { cell: false, tab: 'expression' }, CELLS).expression).toBe('own')
  })

  it('leaves the job without expression when its table is taken away', () => {
    const had = { ...EMPTY_FORM, expression: 'own', own: { ...NO_OWN, expression: table } }
    expect(withOwn(had, NO_OWN, { cell: false, tab: 'expression' }, CELLS).expression).toBe('')
    expect(withOwn({ ...had, expression: 'ENCODE' }, NO_OWN, { cell: false, tab: 'expression' }, CELLS).expression).toBe('ENCODE')
  })
})
