/**
 * One list of jobs on several browsers, joined by a sync code.
 *
 * The server keeps the list. A browser that has the code sends its jobs, takes back the merged list
 * and stores it as its own. A list made without private jobs does not receive one and the private
 * jobs of a browser then stay in that browser.
 */
import { useSyncExternalStore } from 'react'
import { type ApiError, makeList, readList, sendJobs, type SharedList } from '../api'
import { BASE } from '../base'
import { JOBS_EDITED, mergeJobs, recentJobs, replaceJobs } from './recent'
import { ROUTES } from '../routes'

export type Sync = { code: string; private: boolean }
/** `gone` means that the server has no list for the code. */
export type SyncStatus = { state: 'off' | 'syncing' | 'ok' | 'error'; at?: number; error?: string; gone?: boolean }

const KEY = 'gnome3d.sync'
const ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
const LENGTH = 16
const EVERY_MS = 30_000
const FRESH_MS = 5_000

/** Returns the characters of a typed code that can be part of one, with O, I and L read as digits. */
function bare(text: string): string {
  const up = text.toUpperCase().replace(/O/g, '0').replace(/[IL]/g, '1')
  return [...up].filter((ch) => ALPHABET.includes(ch)).join('').slice(0, LENGTH)
}

/** Returns a code in its shown form while it is typed, in groups of four. */
export function formatCode(text: string): string {
  return bare(text).replace(/(.{4})(?=.)/g, '$1-')
}

/** Returns a typed code in its shown form, or null while it is incomplete. */
export function fullCode(text: string): string | null {
  return bare(text).length === LENGTH ? formatCode(text) : null
}

/** Returns the link that opens My jobs on another computer with the code filled in. */
export const joinLink = (code: string) => `${window.location.origin}${BASE}${ROUTES.jobs}?sync=${code}`

export function syncState(): Sync | null {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? 'null')
    return raw && typeof raw.code === 'string' ? { code: raw.code, private: raw.private === true } : null
  } catch {
    return null
  }
}

/** Returns the jobs of this browser that a list takes. */
export const sendable = (list: { private: boolean }) => recentJobs().filter((j) => list.private || !j.url.startsWith('/p/'))

let status: SyncStatus = { state: 'off' }
const watchers = new Set<() => void>()

function setStatus(next: SyncStatus) {
  status = next
  watchers.forEach((w) => w())
}

function adopt(list: SharedList): SharedList {
  localStorage.setItem(KEY, JSON.stringify({ code: list.code, private: list.private }))
  replaceJobs(mergeJobs(recentJobs(), list.jobs))
  setStatus({ state: 'ok', at: Date.now() })
  return list
}

/** Makes a list from the jobs of this browser and joins it. */
export async function createList(includePrivate: boolean): Promise<SharedList> {
  return adopt(await makeList(sendable({ private: includePrivate }), includePrivate))
}

/** Joins the list of a code. The jobs of this browser go to it and its jobs come here. */
export async function joinList(code: string): Promise<SharedList> {
  const list = await readList(fullCode(code) ?? code)
  return adopt(await sendJobs(list.code, sendable(list)))
}

/** Leaves the list. The jobs stay in this browser and the code keeps working elsewhere. */
export function stopSync(): void {
  try {
    localStorage.removeItem(KEY)
  } catch {
    // Nothing was stored.
  }
  setStatus({ state: 'off' })
}

/**
 * Sends the jobs of this browser to its list and stores the merged list.
 * @param keepalive lets the request finish after the page is left, for a job added on submit
 */
export async function syncNow(keepalive = false): Promise<void> {
  const s = syncState()
  if (!s) return setStatus({ state: 'off' })
  setStatus({ ...status, state: 'syncing' })
  // An answer that arrives after the browser left the list is dropped.
  const current = () => syncState()?.code === s.code
  try {
    const list = await sendJobs(s.code, sendable(s), keepalive)
    if (current()) adopt(list)
  } catch (e) {
    if (!current()) return
    const err = e as ApiError
    setStatus({ state: 'error', at: status.at, error: err.message, gone: err.status === 404 })
  }
}

let stop: (() => void) | null = null

/** Keeps the list of this browser in step with the server while the page is open. */
export function startSync(): void {
  if (stop) return
  const edited = () => void syncNow(true)
  const seen = () => {
    if (!document.hidden && Date.now() - (status.at ?? 0) > FRESH_MS) void syncNow()
  }
  const timer = window.setInterval(() => !document.hidden && void syncNow(), EVERY_MS)
  window.addEventListener(JOBS_EDITED, edited)
  window.addEventListener('focus', seen)
  document.addEventListener('visibilitychange', seen)
  stop = () => {
    window.clearInterval(timer)
    window.removeEventListener(JOBS_EDITED, edited)
    window.removeEventListener('focus', seen)
    document.removeEventListener('visibilitychange', seen)
  }
  void syncNow()
}

/** Ends the upkeep of startSync and forgets the status. It is for tests. */
export function resetSync(): void {
  stop?.()
  stop = null
  status = { state: 'off' }
}

const watch = (w: () => void) => {
  watchers.add(w)
  return () => void watchers.delete(w)
}

/** Returns the state of the sync, for a component. */
export const useSyncStatus = () => useSyncExternalStore(watch, () => status)
