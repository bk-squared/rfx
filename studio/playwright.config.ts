import { defineConfig, devices } from "@playwright/test";
import { tmpdir } from "node:os";
import { join } from "node:path";

const workspace =
  process.env.RFX_E2E_WORKSPACE ?? join(tmpdir(), `rfx-studio-playwright-${process.pid}`);
const studioCommand = process.env.RFX_STUDIO_COMMAND ?? "rfx";

export default defineConfig({
  testDir: "./e2e",
  outputDir: "../output/playwright/test-results",
  timeout: 60_000,
  expect: { timeout: 12_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["html", { outputFolder: "../output/playwright/report", open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: `${studioCommand} studio --no-browser --workspace ${JSON.stringify(workspace)} --port 8765`,
      cwd: "..",
      url: "http://127.0.0.1:8765/api/health",
      timeout: 30_000,
      reuseExistingServer: false,
      env: {
        JAX_PLATFORMS: "cpu",
        JAX_PLATFORM_NAME: "cpu",
        CUDA_VISIBLE_DEVICES: "",
      },
    },
    {
      command: "npm run dev",
      cwd: ".",
      url: "http://127.0.0.1:5173",
      timeout: 30_000,
      reuseExistingServer: false,
    },
  ],
});
