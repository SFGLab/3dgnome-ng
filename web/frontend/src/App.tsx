import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Cite } from './pages/Cite'
import { Help } from './pages/Help'
import { Method } from './pages/Method'
import { Model } from './pages/Model'
import { Examples } from './pages/Examples'
import { MyJobs } from './pages/MyJobs'
import { Results } from './pages/Results'
import { Title } from './pages/Title'
import { BASE } from './base'
import { ROUTES } from './routes'
import { startSync } from './sync/sync'

// The explorer carries three.js and loads only on a job page.
const Job = lazy(() => import('./pages/Job').then((m) => ({ default: m.Job })))

/** The site's routes. */
export function App() {
  // A browser with a sync code sends each job it starts or renames, on every page.
  useEffect(() => startSync(), [])
  return (
    <BrowserRouter basename={BASE || '/'}>
      <Layout>
        <Suspense fallback={null}>
        <Routes>
          <Route path={ROUTES.title} element={<Title />} />
          <Route path={ROUTES.cite} element={<Cite />} />
          <Route path={ROUTES.method} element={<Method />} />
          <Route path={ROUTES.results} element={<Results />} />
          <Route path={ROUTES.help} element={<Help />} />
          <Route path={ROUTES.model} element={<Model />} />
          <Route path={ROUTES.jobs} element={<MyJobs />} />
          <Route path={ROUTES.examples} element={<Examples />} />
          <Route path="/jobs/:key" element={<Job kind="jobs" />} />
          <Route path="/p/:key" element={<Job kind="p" />} />
          <Route path="*" element={null} />
        </Routes>
        </Suspense>
      </Layout>
    </BrowserRouter>
  )
}
