import { describe, expect, it } from 'vitest'
import { SCHEMA } from '../test/api'
import { changedSettings, toIni } from './settings'

describe('settings', () => {
  it('keeps only the values that differ from the server', () => {
    expect(changedSettings(SCHEMA, { 'distance.polymer_exponent': '0.3', 'simulation_arcs_smooth.hard_wall': 'yes' })).toEqual({ 'distance.polymer_exponent': '0.3' })
  })

  it('writes one block per section', () => {
    expect(toIni({ 'distance.polymer_exponent': '0.3', 'simulation_arcs_smooth.hard_wall': 'no' })).toBe(
      '[distance]\npolymer_exponent = 0.3\n\n[simulation_arcs_smooth]\nhard_wall = no\n',
    )
  })
})
