import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The UI is served as static files by its own container; /api is proxied to the Backend,
// so the browser always talks to one origin and no CDN is required.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.BACKEND_URL ?? "http://127.0.0.1:8010", changeOrigin: true },
      "/openapi.json": { target: process.env.BACKEND_URL ?? "http://127.0.0.1:8010" },
      "/docs": { target: process.env.BACKEND_URL ?? "http://127.0.0.1:8010" },
    },
  },
  build: { outDir: "dist", sourcemap: false, target: "es2022" },
});
