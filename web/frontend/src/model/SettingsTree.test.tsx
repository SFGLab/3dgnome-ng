import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mockApi, SCHEMA } from '../test/api'
import { checkCopy } from '../test/copy'
import { EnsembleFields, SettingsTree } from './SettingsTree'

function Harness({ initial = {} }: { initial?: Record<string, string> }) {
  const [values, setValues] = useState(initial)
  return <SettingsTree schema={SCHEMA} values={values} onChange={setValues} />
}

describe('SettingsTree', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('filters the settings by name', () => {
    render(<Harness />)
    fireEvent.change(screen.getByPlaceholderText('Filter 3 settings'), { target: { value: 'expon' } })
    const tree = screen.getByRole('tree')
    expect(within(tree).getByText('polymer_exponent')).toBeInTheDocument()
    expect(within(tree).queryByText('hard_wall')).toBeNull()
  })

  it('counts a change on its group and in the head and resets it', () => {
    render(<Harness />)
    fireEvent.change(screen.getByLabelText('polymer_exponent'), { target: { value: '0.3' } })
    expect(within(screen.getByRole('treeitem', { name: /Distance law/ })).getByText('1')).toBeInTheDocument()
    expect(screen.getByText('1 changed')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'default 0.0' }))
    expect(screen.queryByText('1 changed')).toBeNull()
  })

  it('clamps a typed value to the range', () => {
    render(<Harness />)
    const box = screen.getByLabelText('polymer_exponent')
    fireEvent.change(box, { target: { value: '9' } })
    fireEvent.blur(box)
    expect(box).toHaveValue(0.6)
  })

  it('resets every change', () => {
    render(<Harness initial={{ 'distance.polymer_exponent': '0.3', 'distance.contact_half_saturation': '2' }} />)
    expect(screen.getByText('2 changed')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Reset all/ }))
    expect(screen.queryByText(/changed$/)).toBeNull()
  })

  it('reads an uploaded config and lists what it refuses', async () => {
    mockApi([[/^\/api\/settings\/ini$/, () => [200, { overrides: { 'distance.polymer_exponent': '0.2' }, refused: ['data.anchors is not a setting users may change'] }]]])
    render(<Harness />)
    fireEvent.change(screen.getByLabelText(/Upload config.ini/), { target: { files: [new File(['x'], 'c.ini')] } })
    expect(await screen.findByText('data.anchors is not a setting users may change')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('1 changed')).toBeInTheDocument())
  })

  it('keeps to the writing rules', () => {
    const { container } = render(<Harness />)
    expect(checkCopy(container)).toEqual([])
  })
})

describe('EnsembleFields', () => {
  it('takes 1 to the most structures and offers a job name', () => {
    const onChange = vi.fn()
    render(<EnsembleFields structures={100} name="" max={100} onChange={onChange} />)
    const n = screen.getByLabelText('Structures')
    expect(n).toHaveAttribute('min', '1')
    expect(n).toHaveAttribute('max', '100')
    fireEvent.change(n, { target: { value: '250' } })
    fireEvent.blur(n)
    expect(onChange).toHaveBeenLastCalledWith({ structures: 100 })
    expect(screen.getByLabelText('Job name (optional)')).toHaveAttribute('placeholder', 'e.g. TNFRSF8 with RNAPII loops')
  })
})
