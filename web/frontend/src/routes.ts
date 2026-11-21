/** The client routes of the site. */
export const ROUTES = {
  title: '/',
  model: '/model',
  jobs: '/jobs',
  examples: '/examples',
  method: '/method',
  results: '/results',
  help: '/help',
  cite: '/cite',
  privacy: '/privacy',
} as const

/** Parses a job url into its kind and key, or null. */
export function parseJobUrl(text: string): { kind: 'jobs' | 'p'; key: string } | null {
  const m = text.trim().match(/\/(jobs|p)\/([A-Za-z0-9_-]{8,64})/)
  return m ? { kind: m[1] as 'jobs' | 'p', key: m[2] } : null
}
