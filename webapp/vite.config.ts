import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The Hono API/audio server runs on :8787; proxy so the SPA can use same-origin paths.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8787',
      '/audio': 'http://localhost:8787',
    },
  },
});
