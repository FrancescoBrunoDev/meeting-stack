import path from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { viteSingleFile } from "vite-plugin-singlefile"

// One self-contained dashboard.html served by meeting-job: no CDN, no assets.
export default defineConfig({
  plugins: [react(), tailwindcss(), viteSingleFile()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  // `npm run dev` talks to the running service instead of needing a build.
  server: {
    proxy: Object.fromEntries(
      ["/jobs", "/health", "/mcp", "/live"].map((p) => [p, "http://127.0.0.1:8099"])
    ),
  },
})
