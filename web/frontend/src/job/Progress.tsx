import { useEffect, useRef } from 'react'
import type { Summary } from './api'

const WORDS: Record<string, string> = {
  queued: 'Queued',
  inputs: 'Building the inputs',
  hicdiffusion: 'Predicting contacts with HiCDiffusion',
  modelling: 'Modelling the structures',
  marks: 'Marking promoters and enhancers',
  done: 'Done',
}

/** The share of the bar each stage fills; modelling fills its share by structures written. */
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

const since = (s: number) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`)

/** A queued or running job. Over the page: the stage with its count, the bar and the stages with
 * the time each began after the first. Under them the end of the engine log fills the rest of
 * the screen and follows its last line, unless the reader has scrolled up. */
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
  const stages = summary.stages ?? []
  const t0 = stages[0]?.at ?? 0
  const log = useRef<HTMLPreElement>(null)
  const follow = useRef(true)
  useEffect(() => {
    const el = log.current
    if (!el || !follow.current) return
    const toEnd = () => {
      el.scrollTop = el.scrollHeight
    }
    toEnd()
    // Once more after the layout has settled, a late font changes the height of the lines.
    const frame = requestAnimationFrame(toEnd)
    return () => cancelAnimationFrame(frame)
  }, [summary.log_recent])
  return (
    <div className="progress">
      <div className="progress-head">
        <p className="progress-line">{line}</p>
        <span className="progress-pct">{`${pct}%`}</span>
      </div>
      <div className="progress-bar" role="progressbar" aria-label="Progress of the job" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
        <span style={{ width: `${pct}%` }} />
      </div>
      {stages.length > 0 && (
        <ul className="status-log" aria-label="Status log">
          {stages.map((s) => (
            <li key={s.stage}>
              <span>{WORDS[s.stage] ?? s.stage}</span>
              <span className="status-time">{since(s.at - t0)}</span>
            </li>
          ))}
        </ul>
      )}
      <pre
        className="log-tail live"
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
    </div>
  )
}
