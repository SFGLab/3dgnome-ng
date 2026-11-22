import { describe, expect, it } from 'vitest'
import { scoped } from './css'

describe('the class names of a component with CSS modules', () => {
  const cn = scoped({ row: '_row_1a2b3_1', bad: '_bad_1a2b3_9' }, { row: '_row_9z8y7_4' })

  it('gives a name its scoped names and keeps the plain name', () => {
    expect(cn('row')).toBe('_row_1a2b3_1 _row_9z8y7_4 row')
    expect(cn('row bad')).toBe('_row_1a2b3_1 _row_9z8y7_4 row _bad_1a2b3_9 bad')
  })

  it('leaves a name no sheet scopes as it is', () => {
    expect(cn('field-error')).toBe('field-error')
    expect(cn('  row   field-error ')).toBe('_row_1a2b3_1 _row_9z8y7_4 row field-error')
  })

  it('gives nothing for no class', () => {
    expect(cn(undefined)).toBeUndefined()
    expect(cn(false)).toBeUndefined()
    expect(cn('')).toBe('')
  })

  it('gives the same names for names it already scoped', () => {
    expect(cn(cn('row bad'))).toBe(cn('row bad'))
  })
})
