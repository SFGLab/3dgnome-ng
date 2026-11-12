import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Cite } from './pages/Cite'
import { Help } from './pages/Help'
import { Method } from './pages/Method'
import { Model } from './pages/Model'
import { MyJobs } from './pages/MyJobs'
import { Results } from './pages/Results'
import { Title } from './pages/Title'
import { ROUTES } from './routes'

// The explorer carries three.js and loads only on a job page.
const Job = lazy(() => import('./pages/Job').then((m) => ({ default: m.Job })))

/** The site's routes. */
export function App() {
  return (
    <BrowserRouter>
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
          <Route path="/jobs/:key" element={<Job kind="jobs" />} />
          <Route path="/p/:key" element={<Job kind="p" />} />
          <Route path="*" element={null} />
        </Routes>
        </Suspense>
      </Layout>
    </BrowserRouter>
  )
}
