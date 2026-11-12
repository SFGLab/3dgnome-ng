/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // npm run dev serves the frontend and passes the server's own paths to FastAPI on port 8000.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '^/jobs($|/)': 'http://127.0.0.1:8000',
      '/p/': 'http://127.0.0.1:8000',
      '/model': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
