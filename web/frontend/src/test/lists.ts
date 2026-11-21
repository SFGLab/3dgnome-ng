import type { RecentJob } from '../myjobs/recent'
import { mergeJobs } from '../myjobs/recent'
import { mockApi } from './api'

type Stored = { private: boolean; jobs: RecentJob[] }

/** A stand-in for the list API that keeps its lists in memory. Returns the lists and the mock. */
export function mockLists(initial: Record<string, Stored> = {}, newCode = 'AAAA-BBBB-CCCC-DDDD') {
  const lists: Record<string, Stored> = { ...initial }
  const take = (l: Stored, jobs: RecentJob[]) => jobs.filter((j) => l.private || !j.url.startsWith('/p/'))
  const fetch = mockApi([
    [/^\/api\/lists$/, (_u, init) => {
      const body = JSON.parse(String(init?.body))
      lists[newCode] = { private: body.private === true, jobs: [] }
      lists[newCode].jobs = take(lists[newCode], body.jobs)
      return [201, { code: newCode, ...lists[newCode] }]
    }],
    [/^\/api\/lists\/[\w-]+$/, (u, init) => {
      const code = decodeURIComponent(u.pathname.split('/').pop()!)
      const l = lists[code]
      if (!l) return [404, { error: 'No list has this code. Compare it with the computer that shows it.' }]
      if (init?.method === 'PUT') l.jobs = mergeJobs(l.jobs, take(l, JSON.parse(String(init.body)).jobs))
      return [200, { code, ...l }]
    }],
    [/^\/api\/(jobs|p)\/[\w-]+$/, () => [200, { state: 'done', stage: 'done' }]],
  ])
  return { lists, fetch }
}
