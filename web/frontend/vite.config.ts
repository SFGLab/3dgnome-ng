/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// BASE_PATH=/new builds the site for /new/. The server reads the same variable.
const base = (process.env.BASE_PATH ?? '').replace(/\/+$/, '') + '/'

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    // npm run dev serves the frontend and passes the server's own paths to FastAPI on port 8000.
    // With API_ONLY=1 only /api is passed on, and a job page is this build's own. That is the
    // setting for a look at the frontend against a server that was built for another base path.
    proxy: process.env.API_ONLY
      ? { '/api': 'http://127.0.0.1:8000' }
      : {
          '/api': 'http://127.0.0.1:8000',
          '^/jobs($|/)': 'http://127.0.0.1:8000',
          '/p/': 'http://127.0.0.1:8000',
          '/model/classic': 'http://127.0.0.1:8000',
        },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
