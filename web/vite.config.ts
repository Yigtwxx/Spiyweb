import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the FastAPI process so the browser sees one
// origin and no CORS handshake is needed. In production the same bundle is
// served by that FastAPI process, so the origin is genuinely one.
//
// The build lands in `src/spiyweb/viewer/static`, INSIDE the Python package,
// rather than in `web/dist`. That is Faz 2.5: `inspect_url()` has to work
// from an installed wheel, so the bundle is package data now, and building
// into one place rather than building and then copying means the packaged
// bundle can never be a stale copy of the built one. The directory is
// gitignored - a compiled artifact does not belong in a public history - and
// hatchling picks it up through `artifacts` in pyproject.toml.
//
// Sourcemaps are off: they were 972 KB of the 2.6 MB bundle, and a wheel is
// not where anyone debugs minified TypeScript.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false,
        // Server-sent events must not be buffered by the proxy.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (
              proxyRes.headers["content-type"]?.includes("text/event-stream")
            ) {
              proxyRes.headers["cache-control"] = "no-cache";
            }
          });
        },
      },
    },
  },
  build: { outDir: "../src/spiyweb/viewer/static", emptyOutDir: true, sourcemap: false },
});
