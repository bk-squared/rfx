import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api, ApiError } from "./api";
import { CodeEditor } from "./CodeEditor";
import { CopilotPanel } from "./CopilotPanel";
import { buildPatch } from "./diff";
import { EngineeringEvidence } from "./EngineeringEvidence";
import { EngineeringSetup } from "./EngineeringSetup";
import { FieldSlicePlot } from "./FieldSlicePlot";
import { patchGolden } from "./golden";
import { IntentComposer } from "./IntentComposer";
import { S11Plot, SmithChart } from "./RfPlots";
import { SceneViewer } from "./SceneViewer";
import type {
  ExperimentPreview,
  DesignProposal,
  JsonValue,
  RunRecord,
} from "./types";

type WorkspaceTab = "design" | "setup" | "code" | "results";
type SpecDocument = Record<string, JsonValue>;

const recordArray = (value: JsonValue | undefined): Array<Record<string, JsonValue>> =>
  Array.isArray(value)
    ? value.filter((item): item is Record<string, JsonValue> =>
      Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];

const terminalStates = new Set(["succeeded", "failed", "cancelled"]);

const errorText = (error: unknown) => {
  if (error instanceof ApiError) {
    const payload = error.detail as { detail?: { message?: string; code?: string } };
    return [payload.detail?.code, payload.detail?.message].filter(Boolean).join(": ") || error.message;
  }
  return error instanceof Error ? error.message : String(error);
};

const deepCopy = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

export function App() {
  const queryClient = useQueryClient();
  const [selectedExperimentId, setSelectedExperimentId] = useState(
    () => localStorage.getItem("rfx:selected-experiment") ?? "",
  );
  const [selectedRunId, setSelectedRunId] = useState(
    () => localStorage.getItem("rfx:selected-run") ?? "",
  );
  const [tab, setTab] = useState<WorkspaceTab>("design");
  const [draft, setDraft] = useState<SpecDocument | null>(null);
  const [draftText, setDraftText] = useState("");
  const [draftRevisionId, setDraftRevisionId] = useState("");
  const [preview, setPreview] = useState<ExperimentPreview | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [previewLatency, setPreviewLatency] = useState<number | null>(null);
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [exportUrl, setExportUrl] = useState("");
  const [notice, setNotice] = useState("");
  const [agentPanelOpen, setAgentPanelOpen] = useState(false);
  const [copilotPanelOpen, setCopilotPanelOpen] = useState(false);
  const [copilotIntent, setCopilotIntent] = useState("");
  const [copilotProposal, setCopilotProposal] = useState<DesignProposal | null>(null);

  const capabilities = useQuery({
    queryKey: ["capabilities"],
    queryFn: api.getCapabilities,
  });

  const experiments = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 2000,
  });
  const selectedExperiment = experiments.data?.find(
    (item) => item.id === selectedExperimentId,
  );
  const revision = useQuery({
    queryKey: ["revision", selectedExperiment?.current_revision_id],
    queryFn: () => api.getRevision(selectedExperiment!.current_revision_id),
    enabled: Boolean(selectedExperiment),
  });
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: (query) => {
      const records = query.state.data;
      return records?.some((run) => !terminalStates.has(run.state)) ? 500 : 1500;
    },
  });
  const experimentRuns = useMemo(
    () =>
      (runs.data ?? [])
        .filter((run) => run.experiment_id === selectedExperimentId)
        .sort((left, right) => right.created_at.localeCompare(left.created_at)),
    [runs.data, selectedExperimentId],
  );
  const selectedRun =
    experimentRuns.find((run) => run.id === selectedRunId) ?? experimentRuns[0];
  const runRevision = useQuery({
    queryKey: ["run-revision", selectedRun?.revision_id],
    queryFn: () => api.getRevision(selectedRun!.revision_id!),
    enabled: Boolean(selectedRun?.revision_id),
  });
  const s11Artifact = selectedRun?.artifacts.find((item) => item.kind === "s11");
  const s11 = useQuery({
    queryKey: ["s11", s11Artifact?.id],
    queryFn: () => api.readS11(s11Artifact!.url),
    enabled: Boolean(s11Artifact),
  });
  const fieldSliceArtifact = selectedRun?.artifacts.find(
    (item) => item.kind === "field-slice",
  );
  const fieldSlice = useQuery({
    queryKey: ["field-slice", fieldSliceArtifact?.id],
    queryFn: () => api.readFieldSlice(fieldSliceArtifact!.url),
    enabled: Boolean(fieldSliceArtifact),
  });
  const pendingApprovals = useQuery({
    queryKey: ["agent-approvals", "pending"],
    queryFn: () => api.listAgentApprovals("pending"),
    refetchInterval: 1000,
  });
  const agentAudit = useQuery({
    queryKey: ["agent-audit"],
    queryFn: api.listAgentAudit,
    enabled: agentPanelOpen,
    refetchInterval: agentPanelOpen ? 1500 : false,
  });

  useEffect(() => {
    if (selectedExperimentId) localStorage.setItem("rfx:selected-experiment", selectedExperimentId);
  }, [selectedExperimentId]);
  useEffect(() => {
    if (selectedRun?.id && selectedRun.id !== selectedRunId) setSelectedRunId(selectedRun.id);
  }, [selectedRun, selectedRunId]);
  useEffect(() => {
    if (selectedRunId) localStorage.setItem("rfx:selected-run", selectedRunId);
  }, [selectedRunId]);
  useEffect(() => {
    const current = revision.data;
    if (!current || current.id === draftRevisionId) return;
    const next = deepCopy(current.spec);
    setDraft(next);
    setDraftText(JSON.stringify(next, null, 2));
    setDraftRevisionId(current.id);
    setPreview({
      spec_sha256: current.spec_sha256,
      semantic_fingerprint: current.semantic_fingerprint,
      scene: current.scene,
      generated_python: current.generated_python,
      preflight: current.preflight,
      diagnostics: current.diagnostics,
    });
    setPreviewError("");
    setExportUrl("");
  }, [revision.data, draftRevisionId]);

  useEffect(() => {
    if (!draft) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      const started = performance.now();
      try {
        const next = await api.previewExperiment(draft);
        if (!controller.signal.aborted) {
          setPreview(next);
          setPreviewError("");
          setPreviewLatency(performance.now() - started);
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setPreviewError(errorText(error));
          setPreviewLatency(performance.now() - started);
        }
      }
    }, 150);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [draft]);

  const create = useMutation({
    mutationFn: () => api.createExperiment(patchGolden),
    onSuccess: async ({ experiment }) => {
      setSelectedExperimentId(experiment.id);
      setNotice("Patch antenna study created");
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
  const proposeDesign = useMutation({
    mutationFn: ({ intent, revisionId, runId }: { intent: string; revisionId?: string; runId?: string }) =>
      api.proposeDesign(intent, revisionId, runId),
    onMutate: () => {
      setCopilotProposal(null);
      setCopilotPanelOpen(true);
    },
    onSuccess: (proposal) => setCopilotProposal(proposal),
  });
  const createProposedExperiment = useMutation({
    mutationFn: (proposal: DesignProposal) => api.createExperiment(proposal.candidate_spec),
    onSuccess: async ({ experiment }) => {
      setSelectedExperimentId(experiment.id);
      setCopilotPanelOpen(false);
      setCopilotIntent("");
      setNotice("Revision 1 created from the reviewed draft");
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
  const save = useMutation({
    mutationFn: async () => {
      if (!draft || !revision.data || !selectedExperiment) throw new Error("No active revision");
      const patch = buildPatch(revision.data.spec, draft);
      if (!patch.length) throw new Error("No semantic changes to save");
      return api.applyPatch(
        selectedExperiment.id,
        revision.data.id,
        patch,
        "Studio approved semantic diff",
      );
    },
    onSuccess: async (saved) => {
      setDraftRevisionId("");
      setNotice(`Revision ${saved.sequence} created from approved diff`);
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
      await queryClient.invalidateQueries({ queryKey: ["revision"] });
    },
  });
  const validate = useMutation({
    mutationFn: () => api.validateRevision(revision.data!.id),
    onSuccess: (result) =>
      setNotice(result.preflight.ok ? "Preflight passed" : "Preflight blocked the run"),
  });
  const start = useMutation({
    mutationFn: () => api.startRun(selectedExperiment!.id, revision.data!.id),
    onSuccess: async (run) => {
      setSelectedRunId(run.id);
      setTab("results");
      setNotice("CPU run queued in isolated worker");
      await queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
  const cancel = useMutation({
    mutationFn: (run: RunRecord) => api.cancelRun(run.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });
  const exportRun = useMutation({
    mutationFn: () => api.exportRun(selectedRun!.id),
    onSuccess: (artifact) => {
      setExportUrl(artifact.url);
      setNotice("Run bundle is ready to download");
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
  const compare = useMutation({
    mutationFn: (runIds: string[]) => api.compareRuns(runIds),
    onSuccess: () => setNotice("Run comparison loaded"),
  });
  const decideAgentAction = useMutation({
    mutationFn: ({ id, approved }: { id: string; approved: boolean }) =>
      approved ? api.approveAgentAction(id) : api.rejectAgentAction(id),
    onSuccess: async (approval) => {
      setNotice(
        `${approval.tool_name} ${approval.status}; agent may ${approval.status === "approved" ? "retry the exact call" : "not execute it"}`,
      );
      await queryClient.invalidateQueries({ queryKey: ["agent-approvals"] });
      await queryClient.invalidateQueries({ queryKey: ["agent-audit"] });
    },
  });

  const updateDraft = (next: SpecDocument) => {
    setDraft(next);
    setDraftText(JSON.stringify(next, null, 2));
  };
  const updateMetadataTitle = (title: string) => {
    if (!draft) return;
    const next = deepCopy(draft);
    const metadata = next.metadata as Record<string, JsonValue>;
    metadata.title = title;
    updateDraft(next);
  };
  const updateCellSize = (cellSize: number) => {
    if (!draft || !Number.isFinite(cellSize)) return;
    const next = deepCopy(draft);
    const simulation = next.simulation as Record<string, JsonValue>;
    simulation.cell_size_m = cellSize;
    updateDraft(next);
  };
  const updateCenterFrequency = (frequencyGhz: number) => {
    if (!draft || !Number.isFinite(frequencyGhz) || frequencyGhz <= 0) return;
    const next = deepCopy(draft);
    const excitations = recordArray(next.excitations);
    const excitationIndex = excitations.findIndex((item) => Number.isFinite(Number(item.f0_hz)));
    if (excitationIndex < 0) return;
    excitations[excitationIndex].f0_hz = frequencyGhz * 1e9;
    next.excitations = excitations;
    updateDraft(next);
  };
  const updateSweep = (key: "start_hz" | "stop_hz" | "points", value: number) => {
    if (!draft || !Number.isFinite(value) || value <= 0) return;
    const next = deepCopy(draft);
    const observations = recordArray(next.observations);
    const observationIndex = observations.findIndex((item) => item.kind === "sparameters");
    if (observationIndex < 0) return;
    observations[observationIndex][key] = key === "points" ? Math.round(value) : value * 1e9;
    next.observations = observations;
    if (key === "stop_hz") {
      const simulation = next.simulation as Record<string, JsonValue>;
      simulation.freq_max_hz = value * 1e9;
    }
    updateDraft(next);
  };
  const updateDraftText = (text: string) => {
    setDraftText(text);
    try {
      const parsed = JSON.parse(text) as SpecDocument;
      setDraft(parsed);
      setPreviewError("");
    } catch (error) {
      setPreviewError(`invalid_json: ${errorText(error)}`);
    }
  };

  const requestDesignProposal = () => {
    const intent = copilotIntent.trim();
    if (!intent) return;
    proposeDesign.mutate({
      intent,
      revisionId: revision.data?.id,
      runId: selectedRun?.id,
    });
  };

  const useCopilotProposal = () => {
    if (!copilotProposal) return;
    if (!copilotProposal.base_revision_id) {
      createProposedExperiment.mutate(copilotProposal);
      return;
    }
    if (!revision.data || revision.data.id !== copilotProposal.base_revision_id) {
      setNotice("The draft is based on an older revision; request the change again");
      setCopilotPanelOpen(false);
      return;
    }
    const next = deepCopy(copilotProposal.candidate_spec);
    setDraft(next);
    setDraftText(JSON.stringify(next, null, 2));
    setPreview(copilotProposal.preview);
    setPreviewError("");
    setTab("design");
    setCopilotPanelOpen(false);
    setNotice("Design change loaded as an unsaved draft — review it before saving a revision");
  };

  const semanticPatch =
    revision.data && draft ? buildPatch(revision.data.spec, draft) : [];
  const activePreview = preview ?? (revision.data ? {
    spec_sha256: revision.data.spec_sha256,
    semantic_fingerprint: revision.data.semantic_fingerprint,
    scene: revision.data.scene,
    generated_python: revision.data.generated_python,
    preflight: revision.data.preflight,
    diagnostics: revision.data.diagnostics,
  } : null);
  const mutationError =
    create.error ?? createProposedExperiment.error ?? save.error ?? validate.error ?? start.error ?? cancel.error ?? exportRun.error ?? compare.error;
  const copilotProviderLabel = capabilities.data?.design_copilot.llm
    ? `OpenAI · ${capabilities.data.design_copilot.model}`
    : `Offline rules · ${capabilities.data?.design_copilot.model ?? "loading"}`;
  const draftMetadata = draft ? draft.metadata as Record<string, JsonValue> : {};
  const draftSimulation = draft ? draft.simulation as Record<string, JsonValue> : {};
  const draftValidation = draft ? draft.validation as Record<string, JsonValue> : {};
  const primaryExcitation = draft ? recordArray(draft.excitations).find((item) => Number.isFinite(Number(item.f0_hz))) : undefined;
  const sparameterObservation = draft ? recordArray(draft.observations).find((item) => item.kind === "sparameters") : undefined;
  const centerFrequencyGhz = primaryExcitation ? Number(primaryExcitation.f0_hz) / 1e9 : 0;
  const sweepStartGhz = sparameterObservation ? Number(sparameterObservation.start_hz) / 1e9 : 0;
  const sweepStopGhz = sparameterObservation ? Number(sparameterObservation.stop_hz) / 1e9 : 0;
  const sweepPoints = sparameterObservation ? Number(sparameterObservation.points) : 0;
  const fidelityLabel = String(draftMetadata.fidelity ?? "unspecified") === "structural-cpu-smoke"
    ? "CPU screening"
    : String(draftMetadata.fidelity ?? "unspecified");
  const claimLabel = String(draftMetadata.claims ?? "unspecified") === "not-for-quantitative-rf-validation"
    ? "No quantitative RF claim"
    : String(draftMetadata.claims ?? "unspecified");

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Study browser">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">rƒ</span>
          <div><strong>rfx Studio</strong><span>CPU workbench</span></div>
        </div>
        <button className="new-button" onClick={() => create.mutate()} disabled={create.isPending}>
          <span aria-hidden="true">＋</span> New patch study
        </button>
        <nav aria-label="Studies" className="experiment-list">
          <p className="sidebar-label">Studies</p>
          {experiments.isLoading && <p className="muted">Loading studies…</p>}
          {experiments.data?.map((experiment) => (
            <button
              key={experiment.id}
              className={experiment.id === selectedExperimentId ? "experiment active" : "experiment"}
              onClick={() => setSelectedExperimentId(experiment.id)}
            >
              <span className="experiment-icon" aria-hidden="true">⌁</span>
              <span><strong>{experiment.title}</strong><small>Revision {experiment.revision_count}</small></span>
            </button>
          ))}
          {!experiments.isLoading && !experiments.data?.length && (
            <div className="empty-sidebar"><span>∿</span><p>No studies yet</p></div>
          )}
        </nav>
        <div className="system-card">
          <span className="status-dot" />
          <div><strong>Local service</strong><small>127.0.0.1 · CPU only</small></div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="breadcrumb">WORKSPACE / {selectedExperiment ? selectedExperiment.title.toUpperCase() : "START"}</p>
            <h1>{selectedExperiment?.title ?? "New RF study"}</h1>
          </div>
          <div className="topbar-actions">
            <button
              className="copilot-button"
              onClick={() => setCopilotPanelOpen(true)}
              aria-label="Open design assistant"
            >
              <span aria-hidden="true">Δ</span> Design assistant
            </button>
            <button
              className={pendingApprovals.data?.length ? "agent-button pending" : "agent-button"}
              onClick={() => setAgentPanelOpen(true)}
              aria-label={`Tool approvals${pendingApprovals.data?.length ? `, ${pendingApprovals.data.length} pending` : ""}`}
            >
              <span aria-hidden="true">⌾</span> Tool approvals
              {Boolean(pendingApprovals.data?.length) && <b>{pendingApprovals.data?.length}</b>}
            </button>
            {revision.data && (
              <div className="revision-chip" title={revision.data.spec_sha256}>
                <span>REV {revision.data.sequence}</span>
                <code>{revision.data.spec_sha256.slice(0, 8)}</code>
              </div>
            )}
          </div>
        </header>

        {!selectedExperiment ? (
          <section className="welcome" aria-labelledby="welcome-heading">
            <div className="wave-glyph" aria-hidden="true">∿</div>
            <p className="eyebrow">Model · excitation · sweep · outputs</p>
            <h2 id="welcome-heading">Set up an RF simulation</h2>
            <p>Describe the device, materials, excitation, frequency range, and requested outputs. Review the model and CPU estimate before saving the first revision.</p>
            <IntentComposer
              intent={copilotIntent}
              onIntentChange={setCopilotIntent}
              onSubmit={requestDesignProposal}
              pending={proposeDesign.isPending}
              providerLabel={copilotProviderLabel}
            />
            <div className="template-fallback"><span>or</span><button className="secondary" onClick={() => create.mutate()} disabled={create.isPending}>Load patch example</button></div>
            {create.error && <p role="alert" className="error-banner">{errorText(create.error)}</p>}
          </section>
        ) : revision.isLoading || !revision.data || !draft || !activePreview ? (
          <div className="loading-panel">Compiling revision…</div>
        ) : (
          <>
            <div className="toolbar">
              <div className="tabs" role="tablist" aria-label="Study workspace views">
                {(["design", "setup", "code", "results"] as const).map((item) => (
                  <button
                    key={item}
                    role="tab"
                    aria-selected={tab === item}
                    onClick={() => setTab(item)}
                  >
                    {item === "design"
                      ? "Design"
                      : item === "setup"
                        ? "Setup"
                        : item === "code"
                          ? "Model & Code"
                          : `Results${experimentRuns.length ? ` · ${experimentRuns.length}` : ""}`}
                  </button>
                ))}
              </div>
              <div className="toolbar-actions">
                <span className={activePreview.preflight.ok && !previewError ? "gate pass" : "gate fail"}>
                  {activePreview.preflight.ok && !previewError ? "✓ Preflight passed" : "! Preflight blocked"}
                </span>
                <button className="secondary" onClick={() => validate.mutate()} disabled={validate.isPending}>Validate</button>
                <button
                  className="primary"
                  onClick={() => start.mutate()}
                  disabled={!activePreview.preflight.ok || Boolean(previewError) || semanticPatch.length > 0 || start.isPending}
                  title={semanticPatch.length ? "Save the proposed revision before running" : undefined}
                >
                  Run on CPU
                </button>
              </div>
            </div>

            <IntentComposer
              compact
              intent={copilotIntent}
              onIntentChange={setCopilotIntent}
              onSubmit={requestDesignProposal}
              pending={proposeDesign.isPending}
              providerLabel={selectedRun?.state === "succeeded"
                ? `${copilotProviderLabel} · cites run ${selectedRun.id.slice(0, 8)}`
                : copilotProviderLabel}
            />

            {(notice || mutationError) && (
              <div className={mutationError ? "notice error" : "notice"} role={mutationError ? "alert" : "status"}>
                {mutationError ? errorText(mutationError) : notice}
                <button aria-label="Dismiss notification" onClick={() => setNotice("")}>×</button>
              </div>
            )}

            {tab === "design" && (
              <div className="design-grid">
                <section className="panel editor-panel" aria-labelledby="parameters-heading">
                  <header className="panel-heading">
                    <div><p className="eyebrow">Study definition</p><h2 id="parameters-heading">Primary parameters</h2></div>
                    <span className="schema-pill">v2</span>
                  </header>
                  <div className="form-stack">
                    <label>Study name
                      <input
                        aria-label="Study name"
                        value={String(draftMetadata.title)}
                        onChange={(event) => updateMetadataTitle(event.target.value)}
                      />
                    </label>
                    <div className="field-row">
                      <label>Center frequency <div className="input-unit"><input
                        aria-label="Center frequency in GHz"
                        type="number"
                        step="0.01"
                        min="0.01"
                        value={centerFrequencyGhz || ""}
                        onChange={(event) => updateCenterFrequency(event.target.valueAsNumber)}
                      /><span>GHz</span></div></label>
                      <label>Cell size <div className="input-unit"><input
                        aria-label="Cell size in meters"
                        type="number"
                        step="0.0001"
                        value={Number(draftSimulation.cell_size_m)}
                        onChange={(event) => updateCellSize(event.target.valueAsNumber)}
                      /><span>m</span></div></label>
                    </div>
                    <div className="field-row sweep-fields">
                      <label>Sweep start <div className="input-unit"><input
                        aria-label="Sweep start in GHz"
                        type="number"
                        step="0.05"
                        min="0.01"
                        value={sweepStartGhz || ""}
                        onChange={(event) => updateSweep("start_hz", event.target.valueAsNumber)}
                      /><span>GHz</span></div></label>
                      <label>Sweep stop <div className="input-unit"><input
                        aria-label="Sweep stop in GHz"
                        type="number"
                        step="0.05"
                        min="0.01"
                        value={sweepStopGhz || ""}
                        onChange={(event) => updateSweep("stop_hz", event.target.valueAsNumber)}
                      /><span>GHz</span></div></label>
                      <label>Sweep samples <input
                        aria-label="Sweep sample count"
                        type="number"
                        step="1"
                        min="2"
                        value={sweepPoints || ""}
                        onChange={(event) => updateSweep("points", event.target.valueAsNumber)}
                      /></label>
                    </div>
                    <fieldset>
                      <legend>Execution</legend>
                      <div className="readonly-row"><span>Backend</span><strong>CPU</strong></div>
                      <div className="readonly-row"><span>Run class</span><strong>{fidelityLabel}</strong></div>
                    </fieldset>
                  </div>

                  <div className="diff-panel" aria-live="polite">
                    <div className="diff-title"><strong>Pending changes</strong><span>{semanticPatch.length} change{semanticPatch.length === 1 ? "" : "s"}</span></div>
                    {semanticPatch.length ? (
                      <>
                        <ol>{semanticPatch.slice(0, 5).map((operation) => (
                          <li key={`${operation.op}-${operation.path}`}><code>{operation.op}</code><span>{operation.path}</span></li>
                        ))}</ol>
                        <button
                          className="approve-button"
                          onClick={() => save.mutate()}
                          disabled={save.isPending || Boolean(previewError) || !activePreview.preflight.ok}
                        >Save as new revision</button>
                      </>
                    ) : <p className="muted">No unsaved changes.</p>}
                  </div>
                </section>

                <section className="panel geometry-panel" aria-labelledby="geometry-heading">
                  <header className="panel-heading geometry-heading">
                    <div><p className="eyebrow">Model view</p><h2 id="geometry-heading">Geometry</h2></div>
                    <span className="latency">{previewLatency === null ? "up to date" : `${Math.round(previewLatency)} ms`}</span>
                  </header>
                  <SceneViewer scene={activePreview.scene} selected={selectedObject} onSelect={setSelectedObject} />
                  <div className="object-strip" aria-label="Scene objects">
                    {activePreview.scene.entities.map((entity) => (
                      <button
                        key={entity.id}
                        className={selectedObject === entity.id ? "active" : ""}
                        onClick={() => setSelectedObject(entity.id)}
                      ><span style={{ backgroundColor: entity.material_id === "pec" ? "#d09a52" : "#3f936c" }} />{entity.id}</button>
                    ))}
                    {activePreview.scene.overlays.map((overlay) => (
                      <button key={overlay.id} onClick={() => setSelectedObject(overlay.id)}><span className={overlay.role} />{overlay.id}</button>
                    ))}
                  </div>
                </section>

                <section className="panel preflight-panel" aria-labelledby="preflight-heading">
                  <header className="panel-heading">
                    <div><p className="eyebrow">Run checks</p><h2 id="preflight-heading">Preflight</h2></div>
                    <span className={activePreview.preflight.ok && !previewError ? "score good" : "score bad"}>{activePreview.preflight.ok && !previewError ? "PASS" : "BLOCK"}</span>
                  </header>
                  {previewError ? (
                    <button className="issue error-issue" onClick={() => setTab("code")}>
                      <span>!</span><div><strong>Setup is invalid</strong><p>{previewError}</p><small>Open Model & Code to fix the cited field.</small></div>
                    </button>
                  ) : activePreview.preflight.issues.length ? (
                    <div className="issue-list">{activePreview.preflight.issues.map((issue, index) => (
                      <button key={`${issue.code}-${index}`} className={`issue ${issue.severity}`} onClick={() => issue.object_id && setSelectedObject(issue.object_id)}>
                        <span>{issue.severity === "error" ? "!" : "i"}</span><div><strong>{issue.code}</strong><p>{issue.message}</p><small>{issue.path ?? issue.source ?? "simulation"}</small></div>
                      </button>
                    ))}</div>
                  ) : (
                    <div className="preflight-ok"><span>✓</span><div><strong>Preflight passed</strong><p>Model, solver settings, and CPU limits passed the run checks.</p></div></div>
                  )}
                  <div className="gate-facts">
                    <div><span>Support lane</span><strong>{String(draftValidation.support_lane ?? "unspecified")}</strong></div>
                    <div><span>Device</span><strong>CPU / {String(draftSimulation.precision ?? "unspecified")}</strong></div>
                    <div><span>Result use</span><strong>{claimLabel}</strong></div>
                  </div>
                </section>
              </div>
            )}

            {tab === "setup" && (
              <EngineeringSetup
                spec={draft}
                scene={activePreview.scene}
                preflight={activePreview.preflight}
                onOpenSpec={() => setTab("code")}
              />
            )}

            {tab === "code" && (
              <div className="code-grid">
                <section className="panel code-panel" aria-labelledby="spec-heading">
                  <header className="panel-heading"><div><p className="eyebrow">Input model</p><h2 id="spec-heading">ExperimentSpec</h2></div><span className="schema-pill">editable</span></header>
                  <CodeEditor label="ExperimentSpec JSON editor" value={draftText} onChange={updateDraftText} />
                </section>
                <section className="panel code-panel" aria-labelledby="python-heading">
                  <header className="panel-heading"><div><p className="eyebrow">Solver setup</p><h2 id="python-heading">Generated Python</h2></div><span className="schema-pill">read only</span></header>
                  <CodeEditor label="Generated Python" value={activePreview.generated_python} readOnly />
                </section>
                <section className="panel diff-wide" aria-labelledby="code-diff-heading">
                  <header className="panel-heading"><div><p className="eyebrow">Unsaved model edits</p><h2 id="code-diff-heading">Pending changes</h2></div></header>
                  {semanticPatch.length ? <div className="diff-table">{semanticPatch.map((operation) => (
                    <div key={`${operation.op}-${operation.path}`}><code>{operation.op}</code><strong>{operation.path}</strong><span>{operation.op === "remove" ? "removed" : JSON.stringify(operation.value)}</span></div>
                  ))}</div> : <p className="muted padded">No changes from saved revision {revision.data.sequence}.</p>}
                </section>
              </div>
            )}

            {tab === "results" && (
              <div className="results-layout">
                <aside className="run-list panel" aria-label="Run history">
                  <header className="panel-heading"><div><p className="eyebrow">Job queue</p><h2>Runs</h2></div>{experimentRuns.length >= 2 && <button className="compare-button" onClick={() => compare.mutate(experimentRuns.slice(0, 2).map((run) => run.id))}>Compare 2</button>}</header>
                  {experimentRuns.length ? experimentRuns.map((run) => (
                    <button key={run.id} className={run.id === selectedRun?.id ? "run-row active" : "run-row"} onClick={() => setSelectedRunId(run.id)}>
                      <span className={`run-state ${run.state}`} />
                      <span><strong>{run.state}</strong><small>{run.id.slice(0, 8)} · rev {run.revision_id?.slice(0, 6)}</small></span>
                      <time>{new Date(run.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
                    </button>
                  )) : <div className="empty-runs"><span>↗</span><p>No runs yet.</p><small>Validate, then start a CPU run.</small></div>}
                </aside>

                <div className="results-main">
                  {compare.data && (
                    <section className="comparison-card panel" aria-labelledby="comparison-heading">
                      <header className="card-header"><div><p className="eyebrow">Saved runs</p><h2 id="comparison-heading">Run comparison</h2></div><span className="schema-pill">baseline {compare.data.baseline_run_id.slice(0, 6)}</span></header>
                      <div className="comparison-table">
                        <div className="comparison-header"><span>Run</span><span>Min S11</span><span>Resonance</span><span>Δ dB</span><span>Field Δ L2</span><span>Revision state</span></div>
                        {compare.data.rows.map((row) => <div key={row.run_id}><code>{row.run_id.slice(0, 8)}</code><strong>{row.minimum_s11_db.toFixed(2)} dB</strong><span>{(row.minimum_s11_frequency_hz / 1e9).toFixed(2)} GHz</span><span>{row.delta_minimum_s11_db_vs_baseline.toFixed(2)}</span><span>{row.field_normalized_l2_delta_vs_baseline?.toExponential(2) ?? "n/a"}</span><span className="comparison-status">revision {row.validation_status}</span></div>)}
                      </div>
                      <p className="comparison-limitation">{compare.data.limitations.join(" · ")}</p>
                    </section>
                  )}
                  {!selectedRun ? (
                    <section className="panel result-empty"><div className="wave-glyph">∿</div><h2>No run selected</h2><p>Start a CPU run to inspect S-parameters, fields, and run details.</p></section>
                  ) : (
                    <>
                      <section className="run-console panel" aria-labelledby="run-heading">
                        <header className="card-header">
                          <div><p className="eyebrow">CPU solver</p><h2 id="run-heading">Run {selectedRun.id.slice(0, 8)}</h2></div>
                          <div className="run-actions">
                            {!terminalStates.has(selectedRun.state) && <button className="danger" onClick={() => cancel.mutate(selectedRun)}>Cancel</button>}
                            {terminalStates.has(selectedRun.state) && <button className="secondary" onClick={() => exportRun.mutate()} disabled={exportRun.isPending}>Export bundle</button>}
                            {exportUrl && <a className="download-link" href={exportUrl} download>Download .zip</a>}
                          </div>
                        </header>
                        <div className="progress-row"><span>{selectedRun.state}</span><progress value={selectedRun.progress ?? 0} max="1" /><strong>{Math.round((selectedRun.progress ?? 0) * 100)}%</strong></div>
                        {selectedRun.error && <p className="error-banner" role="alert">{selectedRun.error}</p>}
                        <details open>
                          <summary>Event log · {selectedRun.events.length}</summary>
                          <ol className="event-log" tabIndex={0} aria-label="Run event log">{selectedRun.events.slice(-10).reverse().map((event) => (
                            <li key={event.sequence}><code>{String(event.sequence).padStart(2, "0")}</code><span>{event.type}</span><small>{event.state}</small></li>
                          ))}</ol>
                        </details>
                      </section>

                      {selectedRun.state === "succeeded" && s11.data ? (
                        <>
                          <EngineeringEvidence
                            artifact={s11.data}
                            run={selectedRun}
                            revision={runRevision.data}
                            fieldSlice={fieldSlice.data}
                          />
                          <div className="plots-grid"><S11Plot artifact={s11.data} /><SmithChart artifact={s11.data} />
                          {fieldSlice.data ? <FieldSlicePlot artifact={fieldSlice.data} /> : <section className="result-card availability"><header className="card-header"><div><p className="eyebrow">Requested output</p><h3>Field slice</h3></div><span className="availability-tag">loading</span></header><div className="field-placeholder"><span>Ez</span><p>Loading the final field plane from the run artifact.</p></div></section>}
                          <section className="result-card availability"><header className="card-header"><div><p className="eyebrow">Additional outputs</p><h3>Time series & far field</h3></div><span className="availability-tag">not requested</span></header><div className="field-placeholder"><span>t / θ</span><p>No time probe or NTFF surface was requested for this revision.</p></div></section>
                          </div>
                        </>
                      ) : selectedRun.state === "succeeded" ? <div className="loading-panel">Loading S11 results…</div> : null}
                    </>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </main>
      {copilotPanelOpen && (
        <CopilotPanel
          proposal={copilotProposal}
          pending={proposeDesign.isPending || createProposedExperiment.isPending}
          error={proposeDesign.error ? errorText(proposeDesign.error) : createProposedExperiment.error ? errorText(createProposedExperiment.error) : ""}
          onClose={() => setCopilotPanelOpen(false)}
          onUseProposal={useCopilotProposal}
        />
      )}
      {agentPanelOpen && (
        <div className="agent-scrim" role="presentation" onMouseDown={() => setAgentPanelOpen(false)}>
          <aside
            className="agent-panel"
            aria-label="Tool approvals and audit"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <div><p className="eyebrow">External MCP requests</p><h2>Tool approvals</h2></div>
              <button aria-label="Close tool approvals" onClick={() => setAgentPanelOpen(false)}>×</button>
            </header>
            <div className="agent-security-note">
              <span>⌾</span><p><strong>Exact call only</strong>Approval applies once to this actor, tool, and exact argument hash. Any change requires a new approval.</p>
            </div>
            <section aria-labelledby="pending-agent-heading">
              <div className="agent-section-title"><h3 id="pending-agent-heading">Pending approval</h3><span>{pendingApprovals.data?.length ?? 0}</span></div>
              {pendingApprovals.data?.length ? pendingApprovals.data.map((approval) => (
                <article className="approval-card" key={approval.id}>
                  <div className="approval-title"><span>{approval.tool_name.replaceAll("_", " ")}</span><code>{approval.arguments_sha256.slice(0, 10)}</code></div>
                  <p>{approval.actor} · {new Date(approval.requested_at).toLocaleTimeString()}</p>
                  <details open><summary>Semantic impact</summary><pre tabIndex={0}>{JSON.stringify(approval.semantic_diff, null, 2)}</pre></details>
                  <details><summary>Exact arguments</summary><pre tabIndex={0}>{JSON.stringify(approval.arguments, null, 2)}</pre></details>
                  <div className="approval-actions">
                    <button className="reject-action" onClick={() => decideAgentAction.mutate({ id: approval.id, approved: false })}>Reject</button>
                    <button className="approve-action" onClick={() => decideAgentAction.mutate({ id: approval.id, approved: true })}>Approve exact call</button>
                  </div>
                </article>
              )) : <div className="agent-empty"><span>✓</span><p>No tool call is waiting for approval.</p></div>}
            </section>
            <section aria-labelledby="audit-heading">
              <div className="agent-section-title"><h3 id="audit-heading">Tool history</h3><span>{agentAudit.data?.length ?? 0}</span></div>
              <ol className="agent-audit" tabIndex={0}>
                {agentAudit.data?.slice().reverse().slice(0, 20).map((event) => (
                  <li key={event.id}>
                    <span className={`audit-outcome ${event.outcome}`} />
                    <div><strong>{event.tool_name}</strong><small>{event.actor} · {event.outcome}</small></div>
                    <code>#{event.sequence}</code>
                  </li>
                ))}
              </ol>
            </section>
          </aside>
        </div>
      )}
    </div>
  );
}
