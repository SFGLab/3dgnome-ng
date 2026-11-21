import { afterEach, describe, expect, it, vi } from 'vitest'
import { recentJobs, rememberJob, replaceJobs, type RecentJob } from './recent'
import { mockLists } from '../test/lists'
import { createList, formatCode, fullCode, joinLink, joinList, resetSync, startSync, stopSync, syncNow, syncState } from './sync'

const job = (key: string, at: number, kind = 'jobs', name = ''): RecentJob => ({ url: `/${kind}/${key}`, label: `GM12878 ${key}`, name, at })
const urls = (jobs: RecentJob[]) => jobs.map((j) => j.url)

describe('sync codes', () => {
  it('shows a typed code in groups of four', () => {
    expect(formatCode('k7m2')).toBe('K7M2')
    expect(formatCode('k7m29')).toBe('K7M2-9')
    expect(formatCode(' k7m2 9qxd4thp-w3rz ')).toBe('K7M2-9QXD-4THP-W3RZ')
  })

  it('reads O, I and L as the digits they resemble and drops other signs', () => {
    expect(formatCode('oil0_u!x')).toBe('0110-X')
  })

  it('is full at sixteen characters', () => {
    expect(fullCode('K7M2-9QXD-4THP')).toBeNull()
    expect(fullCode('k7m29qxd4thpw3rz')).toBe('K7M2-9QXD-4THP-W3RZ')
  })

  it('links to My jobs with the code', () => {
    expect(joinLink('K7M2-9QXD-4THP-W3RZ')).toBe(`${window.location.origin}/jobs?sync=K7M2-9QXD-4THP-W3RZ`)
  })
})

describe('a synced list', () => {
  afterEach(() => {
    resetSync()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it('is made from the jobs of this browser, without its private jobs', async () => {
    const { lists } = mockLists()
    replaceJobs([job('b', 2, 'p'), job('a', 1)])
    const made = await createList(false)
    expect(made.code).toBe('AAAA-BBBB-CCCC-DDDD')
    expect(syncState()).toEqual({ code: 'AAAA-BBBB-CCCC-DDDD', private: false })
    expect(urls(lists['AAAA-BBBB-CCCC-DDDD'].jobs)).toEqual(['/jobs/a'])
    expect(urls(recentJobs())).toEqual(['/p/b', '/jobs/a'])
  })

  it('takes the private jobs when asked to', async () => {
    const { lists } = mockLists()
    replaceJobs([job('b', 2, 'p'), job('a', 1)])
    await createList(true)
    expect(syncState()?.private).toBe(true)
    expect(urls(lists['AAAA-BBBB-CCCC-DDDD'].jobs)).toEqual(['/p/b', '/jobs/a'])
  })

  it('joins both ways: its jobs come here and the jobs here go to it', async () => {
    const { lists } = mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [job('far', 5)] } })
    replaceJobs([job('near', 3), job('mine', 2, 'p')])
    const list = await joinList('k7m2 9qxd 4thp w3rz')
    expect(list.code).toBe('K7M2-9QXD-4THP-W3RZ')
    expect(urls(recentJobs())).toEqual(['/jobs/far', '/jobs/near', '/p/mine'])
    expect(urls(lists['K7M2-9QXD-4THP-W3RZ'].jobs)).toEqual(['/jobs/far', '/jobs/near'])
  })

  it('does not join a code no list has', async () => {
    mockLists()
    await expect(joinList('0000-0000-0000-0000')).rejects.toThrow('No list has this code')
    expect(syncState()).toBeNull()
  })

  it('sends a job as soon as this browser adds it', async () => {
    const { lists } = mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [] } })
    await joinList('K7M2-9QXD-4THP-W3RZ')
    startSync()
    rememberJob(job('new', 9))
    await vi.waitFor(() => expect(urls(lists['K7M2-9QXD-4THP-W3RZ'].jobs)).toEqual(['/jobs/new']))
    await syncNow()
  })

  it('does not come back when a sync answers after the stop', async () => {
    mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [job('far', 5)] } })
    localStorage.setItem('gnome3d.sync', JSON.stringify({ code: 'K7M2-9QXD-4THP-W3RZ', private: false }))
    const pending = syncNow()
    stopSync()
    await pending
    expect(syncState()).toBeNull()
    expect(recentJobs()).toEqual([])
  })

  it('brings a rename from another computer', async () => {
    const { lists } = mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [] } })
    rememberJob(job('a', 1))
    await joinList('K7M2-9QXD-4THP-W3RZ')
    lists['K7M2-9QXD-4THP-W3RZ'].jobs = [{ ...job('a', 1, 'jobs', 'renamed elsewhere'), edited: Date.now() + 1000 }]
    await syncNow()
    expect(recentJobs()[0].name).toBe('renamed elsewhere')
  })

  it('keeps the jobs and forgets the code on stop', async () => {
    mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [job('far', 5)] } })
    await joinList('K7M2-9QXD-4THP-W3RZ')
    stopSync()
    expect(syncState()).toBeNull()
    expect(urls(recentJobs())).toEqual(['/jobs/far'])
  })
})
