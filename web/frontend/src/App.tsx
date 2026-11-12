import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Cite } from './pages/Cite'
import { Help } from './pages/Help'
import { Method } from './pages/Method'
import { Model } from './pages/Model'
import { Results } from './pages/Results'
import { Title } from './pages/Title'
import { ROUTES } from './routes'

/** The site's routes. Pages are added by the tasks that build them. */
export function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path={ROUTES.title} element={<Title />} />
          <Route path={ROUTES.cite} element={<Cite />} />
          <Route path={ROUTES.method} element={<Method />} />
          <Route path={ROUTES.results} element={<Results />} />
          <Route path={ROUTES.help} element={<Help />} />
          <Route path={ROUTES.model} element={<Model />} />
          <Route path="*" element={null} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
