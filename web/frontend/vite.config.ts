/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'

// BASE_PATH=/new builds the site for /new/. The server reads the same variable.
const base = (process.env.BASE_PATH ?? '').replace(/\/+$/, '') + '/'

// The titles and descriptions of the pages go into the build as pages.json. The server reads the
// file and gives each page its own head.
const pagesJson = {
  name: 'pages-json',
  generateBundle(this: { emitFile: (file: { type: 'asset'; fileName: string; source: string }) => void }) {
    this.emitFile({ type: 'asset', fileName: 'pages.json', source: readFileSync('src/content/pages.json', 'utf8') })
  },
}

export default defineConfig({
  base,
  plugins: [react(), pagesJson],
  server: {
    // npm run dev serves the frontend and passes the paths of the server to FastAPI on port 8000.
    // With API_ONLY=1 only /api is passed on.
    proxy: process.env.API_ONLY
      ? { '/api': 'http://127.0.0.1:8000' }
      : {
          '/api': 'http://127.0.0.1:8000',
          '^/jobs($|/)': 'http://127.0.0.1:8000',
          '/p/': 'http://127.0.0.1:8000',
        },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    // A test finds an element by the plain name of its class. The scoped names of the build are not made.
    css: { modules: { classNameStrategy: 'non-scoped' } },
  },
})
