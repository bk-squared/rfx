export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface ExperimentSummary {
  id: string;
  title: string;
  current_revision_id: string;
  revision_count: number;
  created_at: string;
  updated_at: string;
}

export interface SceneEntity {
  id: string;
  role: "geometry";
  kind: string;
  material_id: string;
  bounds_m: [[number, number, number], [number, number, number]];
}

export interface SceneOverlay {
  id: string;
  role: "excitation" | "observation";
  kind: string;
  position_m?: [number, number, number];
  coordinate_m?: number;
  axis?: "x" | "y" | "z";
}

export interface ScenePreview {
  schema_version: string;
  spec_sha256: string;
  semantic_fingerprint: string;
  workflow: string;
  domain_m: [number, number, number];
  entities: SceneEntity[];
  overlays: SceneOverlay[];
}

export interface PreflightIssue {
  code: string;
  severity: "warning" | "error";
  source?: string;
  message: string;
  path?: string;
  object_id?: string;
}

export interface Revision {
  id: string;
  experiment_id: string;
  sequence: number;
  parent_revision_id: string | null;
  spec_sha256: string;
  semantic_fingerprint: string;
  validation_state: "validated" | "invalid";
  preflight: {
    ok: boolean;
    n_issues: number;
    n_errors: number;
    issues: PreflightIssue[];
  };
  actor: string;
  message: string | null;
  created_at: string;
  spec: Record<string, JsonValue>;
  scene: ScenePreview;
  generated_python: string;
  diagnostics: PreflightIssue[];
}

export interface ExperimentPreview {
  spec_sha256: string;
  semantic_fingerprint: string;
  scene: ScenePreview;
  generated_python: string;
  preflight: Revision["preflight"];
  diagnostics: PreflightIssue[];
}

export interface RunArtifact {
  id: string;
  kind: string;
  sha256: string;
  size_bytes: number;
  url: string;
}

export interface RunEvent {
  sequence: number;
  type: string;
  state: string;
  payload: Record<string, JsonValue> | null;
  created_at: string;
}

export interface RunRecord {
  id: string;
  state: string;
  experiment_id: string | null;
  revision_id: string | null;
  progress: number | null;
  heartbeat_at: string | null;
  artifact_sha256: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  events: RunEvent[];
  artifacts: RunArtifact[];
}

export interface S11Point {
  frequency_hz: number;
  real: number;
  imag: number;
  magnitude_db: number;
}

export interface RuntimeProvenance {
  backend: string;
  devices?: Array<{ platform: string; device_kind: string; id: number }>;
  python_version?: string;
  platform?: string;
  packages?: Record<string, string>;
  source?: {
    git_commit?: string | null;
    git_worktree_dirty?: boolean | null;
    kind?: string;
  };
}

export interface S11Artifact {
  schema_version: string;
  run_id: string;
  spec_sha256: string;
  compiled_sha256: string;
  reference_impedance_ohm: number;
  runtime: RuntimeProvenance;
  points: S11Point[];
}

export interface SParameterValue {
  real: number;
  imag: number;
  magnitude_db: number;
}

export interface SParametersArtifact {
  schema_version: "rfx-sparameters-artifact/v1";
  run_id: string;
  spec_sha256: string;
  compiled_sha256: string;
  port_names: string[];
  runtime: RuntimeProvenance;
  points: Array<{
    frequency_hz: number;
    matrix: SParameterValue[][];
  }>;
}

export interface ReflectionTransmissionArtifact {
  schema_version: "rfx-reflection-transmission-artifact/v1";
  run_id: string;
  spec_sha256: string;
  compiled_sha256: string;
  runtime: RuntimeProvenance;
  points: Array<{
    frequency_hz: number;
    reflection: number;
    transmission: number;
    analytic_reflection: number;
    analytic_transmission: number;
    signal_valid: boolean;
  }>;
}

export interface FieldSliceArtifact {
  schema_version: "rfx-field-slice-artifact/v1";
  run_id: string;
  observation_id: string;
  component: string;
  units: string;
  slice_axis: "x" | "y" | "z";
  requested_coordinate_m: number;
  actual_coordinate_m: number;
  axis_labels: [string, string];
  extent_m: [[number, number], [number, number]];
  shape: [number, number];
  value_encoding: "row-major";
  values: number[][];
  minimum: number;
  maximum: number;
  maximum_absolute: number;
  runtime: { backend: string };
}

export interface JsonPatchOperation {
  op: "add" | "replace" | "remove";
  path: string;
  value?: JsonValue;
}

export interface AgentApproval {
  id: string;
  tool_name: string;
  actor: string;
  arguments_sha256: string;
  arguments: Record<string, JsonValue>;
  semantic_diff: Record<string, JsonValue>;
  status: "pending" | "approved" | "rejected" | "consumed" | "expired";
  requested_at: string;
  expires_at: string;
  decided_at: string | null;
  decided_by: string | null;
}

export interface AgentAuditEvent {
  sequence: number;
  id: string;
  actor: string;
  tool_name: string;
  arguments_sha256: string;
  approval_id: string | null;
  approval_status: string | null;
  outcome: string;
  experiment_id: string | null;
  revision_id: string | null;
  run_id: string | null;
  detail: Record<string, JsonValue>;
  created_at: string;
}

export interface RunComparison {
  schema_version: "rfx-run-comparison/v1";
  analysis_kind: string;
  input_run_ids: string[];
  baseline_run_id: string;
  metric_definition: Record<string, string>;
  rows: Array<{
    run_id: string;
    revision_id: string;
    minimum_s11_db: number;
    minimum_s11_frequency_hz: number;
    max_s11_abs: number;
    delta_minimum_s11_db_vs_baseline: number;
    delta_resonance_hz_vs_baseline: number;
    maximum_field_abs: number | null;
    field_normalized_l2_delta_vs_baseline: number | null;
    validation_status: string;
  }>;
  limitations: string[];
}

export interface StudioCapabilities {
  schema_version: string;
  experiment_schema: string;
  workflows: string[];
  backend: "cpu";
  loopback_only: boolean;
  mcp_endpoint: string;
  design_copilot: {
    schema_version: "rfx-design-proposal/v1";
    provider: string;
    model: string;
    llm: boolean;
    store_provider_responses: boolean;
    mutation_mode: "proposal-only";
    protected_paths: string[];
  };
}

export interface DesignProposal {
  schema_version: "rfx-design-proposal/v1";
  provider: string;
  model: string;
  intent: string;
  workflow: string;
  experiment_id: string | null;
  base_revision_id: string | null;
  answer: string;
  summary: string;
  rationale: string[];
  patch: JsonPatchOperation[];
  expected_effects: string[];
  caveats: string[];
  needs_clarification: boolean;
  question: string;
  candidate_spec: Record<string, JsonValue>;
  preview: ExperimentPreview;
  cpu_estimate: {
    backend: "cpu";
    grid_shape: number[];
    estimated_cells: number;
    estimated_peak_memory_mb: number;
    n_steps: number;
    s_param_n_steps: number;
    estimate_only: boolean;
  };
  run_context: Record<string, JsonValue> | null;
  state_change: "none";
  next_action: "review_then_create" | "review_then_load_draft";
}
