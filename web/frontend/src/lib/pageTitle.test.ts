import { describe, expect, it } from 'vitest'
import { pageTitle } from './pageTitle'

describe('The title of the browser tab', () => {
  it('is the full title on the title page and the page with the tool elsewhere', () => {
    expect(pageTitle('/')).toBe('3D-GNOMEᴺᴳ, 3D chromatin structure from ChIA-PET loops and Hi-C')
    expect(pageTitle('/help')).toBe('Help | 3D-GNOMEᴺᴳ')
    expect(pageTitle('/results/')).toBe('Results | 3D-GNOMEᴺᴳ')
  })

  it('has one title for every job and one for a missing page', () => {
    expect(pageTitle('/jobs/0123456789abcdef')).toBe('Job | 3D-GNOMEᴺᴳ')
    expect(pageTitle('/p/0123456789abcdef')).toBe('Job | 3D-GNOMEᴺᴳ')
    expect(pageTitle('/nothing')).toBe('Page not found | 3D-GNOMEᴺᴳ')
  })
})
