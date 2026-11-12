import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'

/** The site's routes. Pages are added by the tasks that build them. */
export function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="*" element={null} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}
