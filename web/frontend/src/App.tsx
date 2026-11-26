import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { PageTitle } from './components/PageTitle'
import { Cite } from './pages/Cite'
import { Help } from './pages/Help'
import { Method } from './pages/Method'
import { Model } from './pages/Model'
import { Examples } from './pages/Examples'
import { MyJobs } from './pages/MyJobs'
import { NotFound } from './pages/NotFound'
import { Privacy } from './pages/Privacy'
import { Results } from './pages/Results'
import { Title } from './pages/Title'
import { BASE } from './base'
import { hidden } from './flags'
import { ROUTES } from './routes'
import { startSync } from './myjobs/sync'

// The explorer includes three.js and loads only on a job page.
const Job = lazy(() => import('./pages/Job').then((m) => ({ default: m.Job })))

/** The site's routes. */
export function App() {
  // A browser with a sync code sends each job it starts or renames, on every page.
  useEffect(() => startSync(), [])
  return (
    <BrowserRouter basename={BASE || '/'}>
      <PageTitle />
      <Layout>
        <Suspense fallback={null}>
        <Routes>
          <Route path={ROUTES.title} element={<Title />} />
          <Route path={ROUTES.cite} element={<Cite />} />
          {/* A page the server hides leads to the front page. */}
          <Route path={ROUTES.method} element={hidden('method') ? <Navigate to={ROUTES.title} replace /> : <Method />} />
          <Route path={ROUTES.results} element={hidden('results') ? <Navigate to={ROUTES.title} replace /> : <Results />} />
          <Route path={ROUTES.help} element={<Help />} />
          <Route path={ROUTES.model} element={<Model />} />
          <Route path={ROUTES.jobs} element={<MyJobs />} />
          <Route path={ROUTES.examples} element={<Examples />} />
          <Route path={ROUTES.privacy} element={<Privacy />} />
          <Route path="/jobs/:key" element={<Job kind="jobs" />} />
          <Route path="/p/:key" element={<Job kind="p" />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
        </Suspense>
      </Layout>
    </BrowserRouter>
  )
}
