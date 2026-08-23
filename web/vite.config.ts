import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { mockApiPlugin } from './dev/mockApiPlugin.ts'

export default defineConfig({
  plugins: [react(), ...(process.env.DYY_MOCK_API === '1' ? [mockApiPlugin()] : [])],
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/healthz': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    restoreMocks: true,
  },
})
