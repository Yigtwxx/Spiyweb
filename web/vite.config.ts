import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to the FastAPI process so the browser sees one
// origin and no CORS handshake is needed. In production `web/dist` is served
// by that same FastAPI process, so the origin is genuinely one.
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
  build: { outDir: "dist", sourcemap: true },
});
