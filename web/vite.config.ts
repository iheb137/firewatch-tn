import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// En développement, /api est relayé vers la passerelle FastAPI (pas de souci CORS).
const API = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": API } },
  preview: { port: 4173, proxy: { "/api": API } },
  build: { chunkSizeWarningLimit: 1200 },
});
