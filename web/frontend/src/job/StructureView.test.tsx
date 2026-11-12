import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { checkCopy } from '../test/copy'
import { StructureView } from './StructureView'

vi.mock('./Scene', () => ({ Scene: () => <div data-testid="scene" /> }))

const STRUCTURE = {
  chain: [[0, 0, 0, 1000], [1, 0, 0, 2000]] as [number, number, number, number][],
  beads: [[500, 1500, 0, 0, 0, 'anchor'], [1500, 2500, 1, 0, 0, 'subanchor']] as [number, number, number, number, number, 'anchor' | 'subanchor'][],
  marks: { promoters: [], enhancers: [] },
  anchors: [],
}
const ANN = { genes: [], enhancers: [], loops: [] }

function renderView(mode: 'elements' | 'position' | 'factor' = 'elements', onMode = vi.fn()) {
  return render(<StructureView structure={STRUCTURE} annotations={ANN} region={[0, 3000]} index={3} total={100} mode={mode} onMode={onMode} />)
}

describe('StructureView', () => {
  it('draws the scene with the toolbar of the design', () => {
    renderView()
    expect(screen.getByTestId('scene')).toBeInTheDocument()
    for (const name of ['Promoters and enhancers', 'Genomic position', 'Loop factor', 'Reset view', 'Save as PNG', 'Full screen']) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument()
    }
  })

  it('keys each colour mode', () => {
    const { rerender } = renderView('elements')
    expect(screen.getByText('promoters')).toBeInTheDocument()
    expect(screen.getByText('enhancers')).toBeInTheDocument()
    expect(screen.getByText('chromatin')).toBeInTheDocument()
    rerender(<StructureView structure={STRUCTURE} annotations={ANN} region={[0, 3000]} index={3} total={100} mode="factor" onMode={vi.fn()} />)
    expect(screen.getByText('CTCF anchors')).toBeInTheDocument()
    expect(screen.getByText('RNAPII anchors')).toBeInTheDocument()
    rerender(<StructureView structure={STRUCTURE} annotations={ANN} region={[0, 3000]} index={3} total={100} mode="position" onMode={vi.fn()} />)
    expect(screen.getByText('region start')).toBeInTheDocument()
    expect(screen.getByText('region end')).toBeInTheDocument()
  })

  it('asks for another colour mode', () => {
    const onMode = vi.fn()
    renderView('elements', onMode)
    fireEvent.click(screen.getByRole('button', { name: 'Loop factor' }))
    expect(onMode).toHaveBeenCalledWith('factor')
  })

  it('has the caption with the structure number', () => {
    renderView()
    expect(screen.getByText(/^Structure 3 of 100, about one bead per kb\./)).toBeInTheDocument()
  })

  it('keeps to the writing rules', () => {
    const { container } = renderView()
    expect(checkCopy(container)).toEqual([])
  })
})
