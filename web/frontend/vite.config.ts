/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// BASE_PATH=/new builds the site for /new/. The server reads the same variable.
const base = (process.env.BASE_PATH ?? '').replace(/\/+$/, '') + '/'

export default defineConfig({
  base,
  plugins: [react()],
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
  },
})
