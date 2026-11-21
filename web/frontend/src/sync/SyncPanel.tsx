import { Check, Copy, Link2, RefreshCw, Unlink } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useRecentJobs } from '../model/useRecentJobs'
import { createList, formatCode, fullCode, joinLink, joinList, startSync, stopSync, syncNow, syncState, useSyncStatus } from './sync'
import { readList, type SharedList } from '../api'
import '../styles/dialog.css'
import './sync.css'

const clock = new Intl.DateTimeFormat('en-GB', { timeStyle: 'short' })
const jobsText = (n: number) => (n === 1 ? '1 job' : `${n} jobs`)

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    void navigator.clipboard?.writeText(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button type="button" className="sync-button" onClick={copy}>
      {copied ? <Check size={14} aria-hidden /> : label === 'Copy link' ? <Link2 size={14} aria-hidden /> : <Copy size={14} aria-hidden />}
      {copied ? 'Copied' : label}
    </button>
  )
}

/** Asks before this computer leaves the shared list. Escape and Keep syncing close it. */
function StopDialog({ code, onStop, onClose }: { code: string; onStop: () => void; onClose: () => void }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])
  return createPortal(
    <div className="dialog-backdrop">
      <div className="dialog sync-dialog" role="dialog" aria-modal="true" aria-labelledby="sync-stop-title">
        <header className="dialog-head">
          <div>
            <h2 id="sync-stop-title">Stop syncing on this computer</h2>
            <p className="dialog-sub">The jobs stay on this computer and the code keeps working on the others.</p>
          </div>
        </header>
        <div className="dialog-body">
          <p>Keep a copy of the code if no other computer has it.</p>
          <div className="sync-code-row">
            <code className="sync-code">{code}</code>
            <CopyButton text={code} label="Copy code" />
          </div>
        </div>
        <footer className="dialog-foot">
          <button type="button" className="secondary" autoFocus onClick={onClose}>
            Keep syncing
          </button>
          <button type="button" className="danger" onClick={onStop}>
            Stop syncing
          </button>
        </footer>
      </div>
    </div>,
    document.body,
  )
}

/** Makes a code. The one choice is whether the list takes private jobs. */
function CreateForm({ onDone, onCancel }: { onDone: () => void; onCancel: () => void }) {
  const [includePrivate, setIncludePrivate] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const create = () => {
    setBusy(true)
    createList(includePrivate)
      .then(onDone)
      .catch((e: Error) => {
        setError(`${e.message} The code was not made. Try again.`)
        setBusy(false)
      })
  }
  return (
    <div className="sync-form">
      <label className="sync-check">
        <input type="checkbox" checked={includePrivate} onChange={(e) => setIncludePrivate(e.target.checked)} />
        <span>
          <strong>Include private jobs</strong>
          <span className="sync-hint">A private job is one made from uploaded data. With this on, anyone who has the code can open the private jobs on this list. With it off, they stay on this computer.</span>
        </span>
      </label>
      {error && (
        <p className="sync-error" role="alert">
          {error}
        </p>
      )}
      <div className="sync-actions">
        <button type="button" className="sync-button primary" onClick={create} disabled={busy}>
          {busy ? 'Making the code' : 'Create the code'}
        </button>
        <button type="button" className="sync-button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}

/** Takes a code, shows what its list holds and joins it. */
function JoinForm({ initial, onDone, onCancel }: { initial: string; onDone: () => void; onCancel: () => void }) {
  const [text, setText] = useState(formatCode(initial))
  const [found, setFound] = useState<SharedList | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const local = useRecentJobs()
  const code = fullCode(text)

  useEffect(() => {
    setFound(null)
    setError('')
    if (!code) return
    let live = true
    readList(code)
      .then((l) => live && setFound(l))
      .catch((e: Error) => live && setError(e.message))
    return () => {
      live = false
    }
  }, [code])

  const join = () => {
    if (!code) return
    setBusy(true)
    joinList(code)
      .then(onDone)
      .catch((e: Error) => {
        setError(e.message)
        setBusy(false)
      })
  }
  const added = found ? local.filter((j) => (found.private || !j.url.startsWith('/p/')) && !found.jobs.some((f) => f.url === j.url)).length : 0
  return (
    <form
      className="sync-form"
      onSubmit={(e) => {
        e.preventDefault()
        if (found) join()
      }}
    >
      <label className="sync-field">
        <span>Sync code</span>
        <input
          className="sync-input"
          value={text}
          onChange={(e) => setText(formatCode(e.target.value))}
          placeholder="XXXX-XXXX-XXXX-XXXX"
          autoFocus
          autoComplete="off"
          autoCapitalize="characters"
          spellCheck={false}
          aria-describedby="sync-join-note"
        />
      </label>
      <p id="sync-join-note" className="sync-hint" role="status">
        {found
          ? `This list has ${jobsText(found.jobs.length)}. ${found.private ? 'It includes private jobs.' : 'Private jobs are not on it.'}${added ? ` ${jobsText(added)} of this computer will be added to it.` : ''}`
          : error
            ? ''
            : 'Capital and small letters are the same and the dashes are optional.'}
      </p>
      {error && (
        <p className="sync-error" role="alert">
          {error}
        </p>
      )}
      <div className="sync-actions">
        <button type="submit" className="sync-button primary" disabled={!found || busy}>
          {busy ? 'Joining' : 'Join this list'}
        </button>
        <button type="button" className="sync-button" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  )
}

/**
 * The sync of My jobs: make a code, enter one, or see the code this browser has.
 * @param initialCode a code from the page's link, which opens the form to enter it
 * @param onCodeUsed called once the link's code was joined or put aside
 */
export function SyncPanel({ initialCode = '', onCodeUsed }: { initialCode?: string; onCodeUsed?: () => void }) {
  const status = useSyncStatus()
  // The stored state changes only with the status, which every action here sets.
  const sync = syncState()
  const [mode, setMode] = useState<'closed' | 'create' | 'join' | 'stop'>(initialCode && !sync ? 'join' : 'closed')
  const usedLink = useRef(false)

  useEffect(() => startSync(), [])
  useEffect(() => {
    if (initialCode && sync && fullCode(initialCode) === sync.code && !usedLink.current) {
      usedLink.current = true
      onCodeUsed?.()
    }
  }, [initialCode, sync, onCodeUsed])

  const close = () => {
    setMode('closed')
    if (initialCode) onCodeUsed?.()
  }

  if (!sync) {
    return (
      <section className="sync-panel" aria-label="Sync of this list">
        <h2>Your jobs on every computer</h2>
        <p>A sync code links this list with your other computers and browsers. Each of them then shows the same jobs. No account is needed.</p>
        {mode === 'create' ? (
          <CreateForm onDone={close} onCancel={close} />
        ) : mode === 'join' ? (
          <JoinForm initial={initialCode} onDone={close} onCancel={close} />
        ) : (
          <div className="sync-actions">
            <button type="button" className="sync-button primary" onClick={() => setMode('create')}>
              Get a sync code
            </button>
            <button type="button" className="sync-button" onClick={() => setMode('join')}>
              Enter a code
            </button>
          </div>
        )}
      </section>
    )
  }

  const other = initialCode && fullCode(initialCode) !== sync.code
  return (
    <section className="sync-panel synced" aria-label="Sync of this list">
      <div className="sync-head">
        <h2>
          <span className={`sync-dot ${status.state === 'error' ? 'bad' : ''}`} aria-hidden />
          {status.gone ? 'This list is no longer on the server' : status.state === 'error' ? 'The last sync failed' : 'This list is synced'}
        </h2>
        <span className="sync-when" role="status">
          {status.state === 'syncing' ? 'Syncing' : status.at ? `Last synced at ${clock.format(status.at)}` : ''}
        </span>
      </div>
      {status.state === 'error' && (
        <p className="sync-error" role="alert">
          {status.gone ? 'The server has no list for this code. Your jobs are still on this computer. Stop syncing, then get a new code.' : `${status.error} The jobs on this computer are unchanged.`}{' '}
          {!status.gone && (
            <button type="button" className="sync-link" onClick={() => void syncNow()}>
              <RefreshCw size={12} aria-hidden /> Try again
            </button>
          )}
        </p>
      )}
      <div className="sync-code-row">
        <code className="sync-code" aria-label="Sync code">
          {sync.code}
        </code>
        <CopyButton text={sync.code} label="Copy code" />
        <CopyButton text={joinLink(sync.code)} label="Copy link" />
        <button type="button" className="sync-button" aria-haspopup="dialog" onClick={() => setMode('stop')}>
          <Unlink size={14} aria-hidden /> Stop syncing on this computer
        </button>
      </div>
      <p>On another computer, open My jobs, choose Enter a code and type this code. Opening the link there does the same.</p>
      <p>{sync.private ? 'Private jobs are on the shared list. Anyone who has the code can open them.' : 'Private jobs stay on this computer. They are not on the shared list.'}</p>
      {other && (
        <p className="sync-error" role="alert">
          The link you opened has another code. This computer keeps its own code. To use the other one, stop syncing here first.{' '}
          <button type="button" className="sync-link" onClick={() => onCodeUsed?.()}>
            Dismiss
          </button>
        </p>
      )}
      {mode === 'stop' && (
        <StopDialog
          code={sync.code}
          onStop={() => {
            stopSync()
            setMode('closed')
          }}
          onClose={() => setMode('closed')}
        />
      )}
    </section>
  )
}
