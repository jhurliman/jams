import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The Hono API/audio server runs on :8787 (override with API_PORT, e.g. to run a second
// instance side-by-side); proxy so the SPA can use same-origin paths.
const api = `http://localhost:${process.env.API_PORT ?? 8787}`;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': api,
      '/audio': api,
    },
  },
});
