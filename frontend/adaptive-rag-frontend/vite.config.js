import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Real backend (FastAPI etc.) chalne par /api requests yahan forward hongi
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
