import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const BACKEND = process.env.DBV_BACKEND || 'http://127.0.0.1:8077'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: process.env.DBV_HOST || '127.0.0.1',
    proxy: {
      '/api': BACKEND,
      '/img': BACKEND,
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
