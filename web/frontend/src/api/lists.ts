/** The routes of the shared job lists. */
import { call } from './http'
import type { ListedJob, SharedList } from './types'

const json = (method: string, body: unknown, keepalive = false): RequestInit => ({
  method,
  keepalive,
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body),
})
const at = (code: string) => `/api/lists/${encodeURIComponent(code)}`

/** Reads the list of a code without joining it. */
export const readList = (code: string) => call<SharedList>(at(code))

/** Makes a list of jobs on the server and returns its code. */
export const makeList = (jobs: ListedJob[], includePrivate: boolean) => call<SharedList>('/api/lists', json('POST', { jobs, private: includePrivate }))

/**
 * Sends jobs to a list and returns the merged list.
 * @param code the code of the list
 * @param jobs the jobs of this browser
 * @param keepalive lets the request finish after the page is left, for a job added on submit
 */
export const sendJobs = (code: string, jobs: ListedJob[], keepalive = false) => call<SharedList>(at(code), json('PUT', { jobs }, keepalive))
