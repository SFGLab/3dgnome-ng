import { afterEach, describe, expect, it, vi } from 'vitest'
import { recentJobs, rememberJob, type RecentJob } from './recent'

const job = (n: number): RecentJob => ({ url: `/jobs/${n}`, label: `GM12878 chr1:${n}`, name: '', at: n })

describe('recent jobs', () => {
  afterEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it('lists the newest first', () => {
    rememberJob(job(1))
    rememberJob(job(2))
    expect(recentJobs().map((j) => j.url)).toEqual(['/jobs/2', '/jobs/1'])
  })

  it('keeps ten jobs', () => {
    for (let i = 0; i < 12; i++) rememberJob(job(i))
    expect(recentJobs()).toHaveLength(10)
    expect(recentJobs()[0].url).toBe('/jobs/11')
  })

  it('lists a job once', () => {
    rememberJob(job(1))
    rememberJob(job(1))
    expect(recentJobs()).toHaveLength(1)
  })

  it('returns no jobs when the storage throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(recentJobs()).toEqual([])
  })
})
