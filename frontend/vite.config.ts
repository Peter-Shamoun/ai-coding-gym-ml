import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        // Use IPv4 loopback explicitly; Flask is bound on IPv4 and
        // Node on Windows may resolve `localhost` to IPv6 (::1).
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
        timeout: 300000, // 5 min — agent/submit calls can be slow
      },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../static"),
    emptyOutDir: true,
  },
  plugins: [react()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
