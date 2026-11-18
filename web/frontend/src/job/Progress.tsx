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

/** A queued or running job: a bar over its stages, the stage and its count and a status log of
 * the stages with the time each began after the first, then the end of the engine log. */
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
  return (
    <div className="progress">
      <p className="progress-line">{line}</p>
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
      {summary.log_recent && (
        <pre className="log-tail" aria-label="End of the engine log">
          {summary.log_recent}
        </pre>
      )}
      <p className="progress-note">The page refreshes itself and opens the structures when the job is done.</p>
    </div>
  )
}
