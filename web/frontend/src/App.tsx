import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Cite } from './pages/Cite'
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
          <Route path="*" element={null} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
