import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// served from https://marcofer14.github.io/GraphProject/
export default defineConfig({
  base: "/GraphProject/",
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1500 }, // MapLibre alone is ~1 MB (~300 kB gzipped)
});
