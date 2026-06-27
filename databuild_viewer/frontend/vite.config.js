import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// dev：把 /api 和 /img 代理到 FastAPI 后端（同源，免 CORS）
const BACKEND = process.env.DBV_BACKEND || 'http://127.0.0.1:8077'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true, // 监听 0.0.0.0，局域网可访问 dev server
    proxy: {
      '/api': BACKEND,
      '/img': BACKEND,
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
