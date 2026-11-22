import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Unanswered } from './Unanswered'

const asked = { isError: false, error: null }
const refused = { isError: true, error: new Error('The server answered 500.') }

describe('the place of an answer that is not there', () => {
  it('is a busy mark while the question is asked', () => {
    const { container } = render(<Unanswered of={asked} className="view-box" />)
    expect(container.querySelector('.view-box')).toHaveAttribute('aria-busy', 'true')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('has the message of the first question that failed', () => {
    const { container } = render(<Unanswered of={[asked, refused, { isError: true, error: new Error('another') }]} />)
    expect(screen.getByRole('alert')).toHaveTextContent('The server answered 500.')
    expect(container.querySelector('[aria-busy]')).toBeNull()
    expect(checkCopy(container)).toEqual([])
  })
})
