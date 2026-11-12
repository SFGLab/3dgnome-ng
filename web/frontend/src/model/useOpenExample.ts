import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Example } from './examples'
import { openExample } from './openExample'

/** Opens an example's job and reports a refusal of the server. */
export function useOpenExample() {
  const navigate = useNavigate()
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const open = (e: Example) => {
    setBusy(e.id)
    setError('')
    openExample(e)
      .then((to) => navigate(to))
      .catch((err: Error) => setError(err.message))
      .finally(() => setBusy(''))
  }
  return { open, busy, error }
}
