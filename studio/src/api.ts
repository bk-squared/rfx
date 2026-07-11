import type {
  AgentApproval,
  AgentAuditEvent,
  DesignProposal,
  ExperimentSummary,
  ExperimentPreview,
  FieldSliceArtifact,
  JsonPatchOperation,
  Revision,
  RunComparison,
  RunRecord,
  S11Artifact,
  StudioCapabilities,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => response.statusText);
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getCapabilities: () => request<StudioCapabilities>("/api/capabilities"),
  listExperiments: () => request<ExperimentSummary[]>("/api/experiments"),
  createExperiment: (spec: object) =>
    request<{ experiment: ExperimentSummary; revision: Revision }>(
      "/api/experiments",
      { method: "POST", body: JSON.stringify(spec) },
    ),
  previewExperiment: (spec: object) =>
    request<ExperimentPreview>("/api/preview", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  getRevision: (id: string) => request<Revision>(`/api/revisions/${id}`),
  applyPatch: (
    experimentId: string,
    baseRevisionId: string,
    patch: JsonPatchOperation[],
    message: string,
  ) =>
    request<Revision>(`/api/experiments/${experimentId}/patch`, {
      method: "POST",
      body: JSON.stringify({
        base_revision_id: baseRevisionId,
        actor: "human:studio",
        message,
        patch,
      }),
    }),
  validateRevision: (revisionId: string) =>
    request<{
      revision_id: string;
      validation_state: string;
      preflight: Revision["preflight"];
      diagnostics: Revision["diagnostics"];
    }>(`/api/revisions/${revisionId}/validate`, { method: "POST" }),
  listRuns: () => request<RunRecord[]>("/api/runs"),
  startRun: (experimentId: string, revisionId: string) =>
    request<RunRecord>(`/api/experiments/${experimentId}/runs`, {
      method: "POST",
      body: JSON.stringify({
        revision_id: revisionId,
        idempotency_key: `studio-${revisionId}-${crypto.randomUUID()}`,
      }),
    }),
  cancelRun: (runId: string) =>
    request<RunRecord>(`/api/runs/${runId}/cancel`, { method: "POST" }),
  exportRun: (runId: string) =>
    request<{ artifact_id: string; sha256: string; url: string }>(
      `/api/runs/${runId}/export`,
      { method: "POST" },
    ),
  readS11: (url: string) => request<S11Artifact>(url),
  readFieldSlice: (url: string) => request<FieldSliceArtifact>(url),
  compareRuns: (runIds: string[]) =>
    request<RunComparison>("/api/runs/compare", {
      method: "POST",
      body: JSON.stringify({ run_ids: runIds }),
    }),
  listAgentApprovals: (status?: AgentApproval["status"]) =>
    request<AgentApproval[]>(
      `/api/agent/approvals${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ),
  approveAgentAction: (approvalId: string) =>
    request<AgentApproval>(`/api/agent/approvals/${approvalId}/approve`, {
      method: "POST",
      body: JSON.stringify({ actor: "human:studio" }),
    }),
  rejectAgentAction: (approvalId: string) =>
    request<AgentApproval>(`/api/agent/approvals/${approvalId}/reject`, {
      method: "POST",
      body: JSON.stringify({ actor: "human:studio" }),
    }),
  listAgentAudit: () => request<AgentAuditEvent[]>("/api/agent/audit?limit=100"),
  proposeDesign: (intent: string, revisionId?: string, runId?: string) =>
    request<DesignProposal>("/api/copilot/proposals", {
      method: "POST",
      body: JSON.stringify({
        intent,
        revision_id: revisionId || undefined,
        run_id: runId || undefined,
      }),
    }),
};
