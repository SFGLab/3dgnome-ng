import { duration } from '../lib/format'
import { Check } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Summary } from '../api'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

const WORDS: Record<string, string> = {
  queued: 'Queued',
  inputs: 'Building the inputs',
  hicdiffusion: 'Predicting contacts with HiCDiffusion',
  modelling: 'Modelling the structures',
  marks: 'Marking promoters and enhancers',
  done: 'Done',
}

/** Returns the share of the bar each stage fills. Modelling fills its share by structures written. */
function weights(hicdiffusion: boolean): [string, number][] {
  return hicdiffusion
    ? [
        ['inputs', 5],
        ['hicdiffusion', 10],
        ['modelling', 75],
        ['marks', 10],
      ]
    : [
        ['inputs', 5],
        ['modelling', 85],
        ['marks', 10],
      ]
}


/**
 * A queued or running job. It has the stage with its count, the bar and the stages, then the end of
 * the engine log, which follows its last line unless the reader has scrolled up. The stages are the
 * ones the job goes through, in order. A finished stage has a tick and the time it took and the
 * running one a spinner and its time so far. A stage to come has neither. A waiting job has no
 * engine yet and no log.
 */
export function Progress({ summary }: { summary: Summary }) {
  const stage = summary.state === 'queued' ? 'queued' : (summary.stage ?? 'inputs')
  const ws = weights((summary.request.background ?? '') === 'hicdiffusion' || !!summary.request.hicdiffusion)
  const done = summary.structures_done ?? 0
  const asked = summary.structures_asked ?? summary.request.n_models
  let pct = 0
  for (const [name, w] of ws) {
    if (name === stage) {
      if (name === 'modelling' && asked) pct += (w * Math.min(done, asked)) / asked
      break
    }
    if (stage !== 'queued') pct += w
  }
  pct = Math.round(pct)
  const line =
    stage === 'queued'
      ? summary.ahead
        ? `Waiting in the queue behind ${summary.ahead} job${summary.ahead === 1 ? '' : 's'}`
        : 'Waiting in the queue, next to run'
      : stage === 'modelling' && asked
        ? `${WORDS.modelling}, ${done} of ${asked} written`
        : (WORDS[stage] ?? stage)
  const reached = summary.stages ?? []
  // The time of the running stage is counted on the clock of the server, from its answer on.
  const got = useMemo(() => ({ at: Date.now(), now: summary.now ?? Date.now() / 1000 }), [summary])
  const [, setTick] = useState(0)
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [])
  const now = got.now + (Date.now() - got.at) / 1000
  const plan = ['queued', ...ws.map(([name]) => name)]
  const rows = plan.map((name) => {
    const i = reached.findIndex((s) => s.stage === name)
    if (name === stage) return { name, state: 'now' as const, took: i >= 0 ? Math.max(0, now - reached[i].at) : null }
    // A stage is over once a later one began. It took the time up to that beginning.
    if (i >= 0 && i + 1 < reached.length) return { name, state: 'done' as const, took: reached[i + 1].at - reached[i].at }
    return { name, state: 'next' as const, took: null }
  })
  const log = useRef<HTMLPreElement>(null)
  const follow = useRef(true)
  useEffect(() => {
    const el = log.current
    if (!el || !follow.current) return
    const toEnd = () => {
      el.scrollTop = el.scrollHeight
    }
    toEnd()
    // Once more after the layout has settled and once the log's font has loaded, which changes
    // the height of its lines.
    const frame = requestAnimationFrame(toEnd)
    void document.fonts?.ready.then(() => {
      if (follow.current) toEnd()
    })
    return () => cancelAnimationFrame(frame)
  }, [summary.log_recent])
  return (
    <div className={cn(stage === 'queued' ? 'progress waiting' : 'progress')}>
      <div className={cn('progress-head')}>
        <p className={cn('progress-line')}>{line}</p>
        <span className={cn('progress-pct')}>{`${pct}%`}</span>
      </div>
      <div className={cn('progress-bar')} role="progressbar" aria-label="Progress of the job" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
        <span style={{ width: `${pct}%` }} />
      </div>
      <ol className={cn('status-log')} aria-label="Stages of the job">
        {rows.map((r) => (
          <li key={r.name} className={cn(r.state)} aria-current={r.state === 'now' ? 'step' : undefined}>
            {r.state === 'done' ? <Check className={cn('stage-mark')} size={13} aria-hidden /> : <span className={cn(r.state === 'now' ? 'stage-mark stage-spinner' : 'stage-mark stage-dot')} aria-hidden />}
            <span>{WORDS[r.name] ?? r.name}</span>
            {r.took !== null && <span className={cn('status-time')}>{duration(r.took)}</span>}
          </li>
        ))}
      </ol>
      {stage !== 'queued' && (
        <pre
          className={cn('log-tail live')}
          aria-label="End of the engine log"
          ref={log}
          onScroll={(e) => {
            const el = e.currentTarget
            // The log follows its end while the reader is at the end.
            follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
          }}
        >
          {summary.log_recent || 'The engine has not written to its log yet.'}
        </pre>
      )}
    </div>
  )
}
