/** The jobs of this browser, kept in localStorage, newest first. */
export type RecentJob = { url: string; label: string; name: string; at: number; edited?: number }

const KEY = 'gnome3d.recentJobs'
// Enough for a working session. Older jobs stay reachable by their links.
export const MAX = 50

/** Fired on the window after the stored list changed, by this page or by a sync. */
export const JOBS_CHANGED = 'gnome3d:jobs-changed'
/** Fired on the window after this page added or renamed a job. */
export const JOBS_EDITED = 'gnome3d:jobs-edited'

export function recentJobs(): RecentJob[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]')
    return Array.isArray(raw) ? raw : []
  } catch {
    return []
  }
}

const stamp = (j: RecentJob) => j.edited ?? j.at

/** Returns the name this browser gave a job, empty without one. */
export const nameOf = (url: string) => recentJobs().find((j) => j.url === url)?.name ?? ''

/** Returns two lists as one, newest first. A job on both keeps the entry edited last. */
export function mergeJobs(a: RecentJob[], b: RecentJob[]): RecentJob[] {
  const out = new Map(a.map((j) => [j.url, j]))
  for (const j of b) {
    const have = out.get(j.url)
    if (!have || stamp(j) > stamp(have)) out.set(j.url, j)
  }
  return [...out.values()].sort((x, y) => y.at - x.at).slice(0, MAX)
}

/** Stores a list as the jobs of this browser. Returns false in a browser without storage. */
export function replaceJobs(jobs: RecentJob[]): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(jobs.slice(0, MAX)))
  } catch {
    return false
  }
  window.dispatchEvent(new Event(JOBS_CHANGED))
  return true
}

/** Adds a job to the list, or replaces its entry and marks the entry as edited now. */
export function rememberJob(job: RecentJob): void {
  const rest = recentJobs().filter((j) => j.url !== job.url)
  if (replaceJobs(mergeJobs([{ ...job, edited: Date.now() }], rest))) window.dispatchEvent(new Event(JOBS_EDITED))
}
