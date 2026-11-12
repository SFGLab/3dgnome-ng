/** The jobs this browser submitted, kept in localStorage, newest first. */
export type RecentJob = { url: string; label: string; name: string; at: number }

const KEY = 'gnome3d.recentJobs'
const MAX = 10

export function recentJobs(): RecentJob[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]')
    return Array.isArray(raw) ? raw : []
  } catch {
    return []
  }
}

export function rememberJob(job: RecentJob): void {
  try {
    const rest = recentJobs().filter((j) => j.url !== job.url)
    localStorage.setItem(KEY, JSON.stringify([job, ...rest].slice(0, MAX)))
  } catch {
    // A browser without storage keeps no list.
  }
}
