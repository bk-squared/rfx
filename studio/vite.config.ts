import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const stripGeneratedTrailingWhitespace = (): Plugin => ({
  name: "strip-generated-trailing-whitespace",
  generateBundle(_options, bundle) {
    for (const output of Object.values(bundle)) {
      if (output.type === "chunk") output.code = output.code.replace(/[ \t]+$/gm, "");
    }
  },
});

export default defineConfig({
  plugins: [react(), stripGeneratedTrailingWhitespace()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/mcp": "http://127.0.0.1:8765",
    },
  },
  build: {
    outDir: "../rfx/studio/static",
    emptyOutDir: true,
    sourcemap: false,
  },
});
