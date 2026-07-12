import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const wr90Spec = JSON.parse(
  readFileSync(new URL("../../rfx/studio/templates/wr90_waveguide.json", import.meta.url), "utf8"),
);

test("patch journey: create, edit, preview, run, inspect, compare, cancel, export", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Load patch example" }).click();
  await expect(page.getByRole("heading", { name: "2.4 GHz FR4 patch antenna" })).toBeVisible();
  await expect(page.getByTestId("scene-viewer")).toBeVisible();
  await expect(page.locator(".latency")).not.toHaveText("canonical");
  await page
    .locator(".object-strip")
    .getByRole("button", { name: "patch", exact: true })
    .click();
  await expect(
    page
      .locator(".object-strip")
      .getByRole("button", { name: "patch", exact: true }),
  ).toHaveClass(/active/);
  await expect(page.getByLabel("Center frequency in GHz")).toHaveValue("2.4");
  await expect(page.getByLabel("Sweep start in GHz")).toHaveValue("1.8");
  await expect(page.getByLabel("Sweep stop in GHz")).toHaveValue("3");
  await expect(page.getByLabel("Sweep sample count")).toHaveValue("9");

  await page.getByLabel("Cell size in meters").fill("0");
  await expect(page.getByRole("button", { name: "Run on CPU" })).toBeDisabled();
  await expect(page.getByText("Setup is invalid", { exact: true })).toBeVisible();
  await page.getByText("Setup is invalid", { exact: true }).click();
  await expect(page.getByRole("tab", { name: "Model & Code" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.getByRole("tab", { name: "Design" }).click();
  await page.getByLabel("Cell size in meters").fill("0.0015");
  await page.getByLabel("Center frequency in GHz").fill("2.45");
  await page.getByLabel("Sweep sample count").fill("11");
  await expect(page.getByText("✓ Preflight passed", { exact: true })).toBeVisible();

  const previewStarted = Date.now();
  const previewResponse = page.waitForResponse(
    (response) => response.url().endsWith("/api/preview") && response.request().method() === "POST",
  );
  await page.getByLabel("Study name").fill("2.45 GHz patch sweep study");
  expect((await previewResponse).ok()).toBeTruthy();
  expect(Date.now() - previewStarted).toBeLessThan(1000);
  await expect(page.getByText("4 changes", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Save as new revision" }).click();
  await expect(page.getByText("Revision 2 created", { exact: false })).toBeVisible();
  await expect(page.getByRole("heading", { name: "2.45 GHz patch sweep study" })).toBeVisible();

  await page.getByRole("tab", { name: "Setup" }).click();
  await expect(page.getByRole("heading", { name: "Simulation setup" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Domain & mesh" })).toBeVisible();
  await expect(page.getByText("36 × 33 × 31", { exact: true })).toBeVisible();
  await expect(page.getByText("36,828", { exact: true })).toBeVisible();
  await expect(page.getByText("Screening run only", { exact: true })).toBeVisible();
  await expect(page.getByText("1.1 cells / min feature", { exact: true })).toBeVisible();
  await expect(page.getByText("11 points are enough for a coarse sweep", { exact: false })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Ports & excitations" })).toBeVisible();
  await expect(page.getByText("50.0 Ω", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Boundary conditions" })).toBeVisible();
  await expect(page.getByText("4 CPML layers", { exact: true })).toBeVisible();

  await page.getByRole("tab", { name: "Model & Code" }).click();
  await expect(page.getByRole("heading", { name: "Generated Python" })).toBeVisible();
  await expect(page.locator('[aria-label="Generated Python"]')).toContainText(
    "build_simulation",
  );
  await page.getByRole("tab", { name: "Design" }).click();
  await page.getByRole("tab", { name: "Design" }).press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Setup" })).toBeFocused();
  await expect(page.getByRole("tab", { name: "Setup" })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("tab", { name: "Setup" }).press("Home");
  await expect(page.getByRole("tab", { name: "Design" })).toBeFocused();

  await page.getByRole("button", { name: "Validate", exact: true }).click();
  await expect(page.getByText("✓ Preflight passed", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Run on CPU" }).click();
  await expect(page.getByRole("tab", { name: /Results/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("heading", { name: /Run / })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "2.45 GHz patch sweep study" })).toBeVisible();
  await page.getByRole("tab", { name: /Results/ }).click();
  await expect(page.getByText("succeeded", { exact: true }).first()).toBeVisible({ timeout: 45_000 });
  await expect(page.getByRole("heading", { name: "S11 magnitude" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Smith chart" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "EZ field slice" })).toBeVisible();
  await expect(page.getByTestId("field-slice-heatmap")).toBeVisible();
  const evidence = page.locator(".engineering-evidence");
  await expect(evidence.getByRole("heading", { name: "Run summary" })).toBeVisible();
  await expect(evidence.getByText("VSWR at minimum", { exact: true })).toBeVisible();
  await expect(evidence.getByText("Convergence trace", { exact: true })).toBeVisible();
  await expect(evidence.getByText("not recorded", { exact: true })).toHaveCount(3);
  await expect(evidence.getByText("Spec SHA", { exact: true })).toBeVisible();
  await expect(evidence.getByText("Sweep boundary", { exact: true })).toBeVisible();
  await expect(evidence.getByText("Sample density", { exact: true })).toBeVisible();
  await expect(evidence.getByText("Field plane", { exact: true })).toBeVisible();
  await expect(evidence.getByText("120.0 MHz step", { exact: true })).toBeVisible();
  await expect(evidence.getByText("Validation checks", { exact: true })).toBeVisible();
  await expect(evidence.getByText("no recorded result", { exact: true })).toHaveCount(2);
  await expect(page.getByText("Nearest solved plane · Δ 0.500 mm", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Time series & far field" })).toBeVisible();
  await expect(evidence.getByText("No quantitative RF claim", { exact: false })).toBeVisible();

  await page.getByRole("button", { name: "Export bundle" }).click();
  const downloadLink = page.getByRole("link", { name: "Download .zip" });
  await expect(downloadLink).toBeVisible();
  const downloadPromise = page.waitForEvent("download");
  await downloadLink.click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/rfx-replay-.*\.zip/);

  await page.reload();
  await expect(page.getByRole("heading", { name: "2.45 GHz patch sweep study" })).toBeVisible();
  await page.getByRole("tab", { name: /Results/ }).click();
  await expect(page.getByText("succeeded", { exact: true }).first()).toBeVisible();

  const accessibility = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    accessibility.violations.filter((violation) =>
      ["serious", "critical"].includes(violation.impact ?? ""),
    ),
  ).toEqual([]);

  await page.getByRole("button", { name: "Run on CPU" }).click();
  await expect(page.locator(".run-row .run-state.succeeded")).toHaveCount(2, {
    timeout: 45_000,
  });
  await page.getByRole("button", { name: "Compare 2" }).click();
  await expect(page.getByRole("heading", { name: "Run comparison" })).toBeVisible();
  await expect(page.locator(".comparison-table").getByText("revision validated", { exact: true })).toHaveCount(2);

  const experiments = await (await page.request.get("/api/experiments")).json();
  const experiment = experiments.find(
    (candidate: { title: string }) => candidate.title === "2.45 GHz patch sweep study",
  );
  expect(experiment).toBeTruthy();
  const currentRevision = await (
    await page.request.get(`/api/revisions/${experiment.current_revision_id}`)
  ).json();
  const longRevision = await page.request.post(
    `/api/experiments/${experiment.id}/patch`,
    {
      data: {
        base_revision_id: currentRevision.id,
        actor: "human:e2e",
        message: "Create a deterministic cancellation witness",
        patch: [
          { op: "replace", path: "/execution/n_steps", value: 500_000 },
          {
            op: "replace",
            path: "/execution/s_param_n_steps",
            value: 500_000,
          },
        ],
      },
    },
  );
  expect(longRevision.status()).toBe(201);
  await page.reload();
  await expect(page.getByRole("heading", { name: "2.45 GHz patch sweep study" })).toBeVisible();
  await page.getByRole("tab", { name: /Results/ }).click();
  await page.getByRole("button", { name: "Run on CPU" }).click();
  const runningRun = page
    .locator("button.run-row")
    .filter({ has: page.locator(".run-state.running") });
  await expect(runningRun).toHaveCount(1, { timeout: 15_000 });
  await runningRun.click();
  await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible({
    timeout: 15_000,
  });
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.locator(".run-row .run-state.cancelled")).toHaveCount(1, {
    timeout: 20_000,
  });
});

test("agent patch is inert until Studio approves the exact MCP call", async ({ page }) => {
  const experiments = await (await page.request.get("/api/experiments")).json();
  expect(experiments.length).toBeGreaterThan(0);
  const experiment = experiments[0];
  const revision = await (
    await page.request.get(`/api/revisions/${experiment.current_revision_id}`)
  ).json();
  const argumentsPayload = {
    experiment_id: experiment.id,
    base_revision_id: revision.id,
    patch: [
      {
        op: "replace",
        path: "/metadata/title",
        value: "MCP-approved patch antenna",
      },
    ],
    message: "OpenAI/Claude-neutral MCP proposal",
    actor: "agent:openai-e2e",
  };
  const rpc = async (id: number, args: Record<string, unknown>) => {
    const response = await page.request.post("http://127.0.0.1:8765/mcp/", {
      headers: { Accept: "application/json, text/event-stream" },
      data: {
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name: "apply_experiment_patch", arguments: args },
      },
    });
    expect(response.ok()).toBeTruthy();
    return response.json();
  };

  const proposed = await rpc(1, argumentsPayload);
  expect(proposed.result.structuredContent.status).toBe("approval_required");
  const approvalId = proposed.result.structuredContent.approval.id;

  await page.goto("/");
  const agentButton = page.getByRole("button", { name: "Tool approvals, 1 pending" });
  await expect(agentButton).toBeVisible();
  await agentButton.click();
  await expect(page.getByRole("heading", { name: "Tool approvals" })).toBeVisible();
  await expect(page.getByText("apply experiment patch", { exact: true })).toBeVisible();
  await expect(page.locator(".approval-card details[open] pre")).toContainText(
    "/metadata/title",
  );

  const panelAccessibility = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    panelAccessibility.violations.filter((violation) =>
      ["serious", "critical"].includes(violation.impact ?? ""),
    ),
  ).toEqual([]);

  await page.getByRole("button", { name: "Approve exact call" }).click();
  await expect(page.getByText("No tool call is waiting for approval.")).toBeVisible();
  const executed = await rpc(2, { ...argumentsPayload, approval_id: approvalId });
  expect(executed.result.structuredContent.status).toBe("ok");
  await page.getByRole("button", { name: "Close tool approvals" }).click();

  await expect(page.getByText("MCP-approved patch antenna", { exact: true })).toBeVisible();
});

test("WR-90 primary controls keep coupled ports and sweep values synchronized", async ({ page }) => {
  const created = await page.request.post("/api/experiments", { data: wr90Spec });
  expect(created.status(), await created.text()).toBe(201);

  await page.goto("/");
  await page.getByRole("button", { name: /WR-90 empty matched guide/ }).click();
  await expect(page.getByLabel("Center frequency in GHz")).toHaveValue("10.3");
  await expect(page.getByLabel("Sweep start in GHz")).toHaveValue("8.2");
  await expect(page.getByLabel("Sweep stop in GHz")).toHaveValue("12.4");
  await expect(page.getByLabel("Sweep sample count")).toHaveValue("5");

  await page.getByLabel("Center frequency in GHz").fill("10.5");
  await page.getByLabel("Sweep stop in GHz").fill("12.6");
  await page.getByLabel("Sweep sample count").fill("17");
  await expect(page.getByRole("button", { name: "Save as new revision" })).toBeEnabled();
  await page.getByRole("button", { name: "Save as new revision" }).click();
  await expect(page.getByText("Revision 2 created", { exact: false })).toBeVisible();

  const experiments = await (await page.request.get("/api/experiments")).json();
  const wr90 = experiments.find(
    (candidate: { title: string }) => candidate.title === "WR-90 empty matched guide",
  );
  const revision = await (
    await page.request.get(`/api/revisions/${wr90.current_revision_id}`)
  ).json();
  expect(revision.spec.excitations.map((item: { f0_hz: number }) => item.f0_hz)).toEqual([
    10.5e9,
    10.5e9,
  ]);
  expect(revision.spec.excitations.map((item: { stop_hz: number }) => item.stop_hz)).toEqual([
    12.6e9,
    12.6e9,
  ]);
  expect(revision.spec.excitations.map((item: { points: number }) => item.points)).toEqual([
    17,
    17,
  ]);
});
