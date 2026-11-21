import { useEffect, useState } from 'react'
import { JOBS_CHANGED, recentJobs, type RecentJob } from './recent'

/** Returns the jobs of this browser, read again after every change of the stored list. */
export function useRecentJobs(): RecentJob[] {
  const [jobs, setJobs] = useState(recentJobs)
  useEffect(() => {
    const read = () => setJobs(recentJobs())
    window.addEventListener(JOBS_CHANGED, read)
    return () => window.removeEventListener(JOBS_CHANGED, read)
  }, [])
  return jobs
}
