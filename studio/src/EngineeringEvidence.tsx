import type { Revision, RunRecord, S11Artifact } from "./types";

interface EngineeringEvidenceProps {
  artifact: S11Artifact;
  run: RunRecord;
  revision?: Revision;
  hasFieldSlice: boolean;
}

const formatFrequency = (value: number) => `${(value / 1e9).toFixed(3)} GHz`;

const shortHash = (value: string | undefined | null) => value ? value.slice(0, 10) : "not available";

const duration = (run: RunRecord) => {
  const terminalEvent = [...run.events].reverse().find((event) =>
    ["run_succeeded", "run_failed", "run_cancelled"].includes(event.type));
  const finishedAt = terminalEvent?.created_at ?? run.updated_at;
  const elapsed = new Date(finishedAt).getTime() - new Date(run.created_at).getTime();
  if (!Number.isFinite(elapsed) || elapsed < 0) return "not available";
  if (elapsed < 1000) return `${elapsed} ms`;
  return `${(elapsed / 1000).toFixed(1)} s`;
};

export function EngineeringEvidence({ artifact, run, revision, hasFieldSlice }: EngineeringEvidenceProps) {
  const points = [...artifact.points].sort((left, right) => left.frequency_hz - right.frequency_hz);
  if (!points.length) {
    return (
      <section className="panel engineering-evidence" aria-labelledby="rf-evidence-heading">
        <header className="panel-heading">
          <div><p className="eyebrow">Engineering interpretation</p><h2 id="rf-evidence-heading">RF evidence summary</h2></div>
          <span className="schema-pill">incomplete artifact</span>
        </header>
        <p className="evidence-boundary">No S11 samples were captured. Engineering metrics cannot be inferred from an empty artifact.</p>
      </section>
    );
  }
  const minimum = points.reduce((best, point) =>
    point.magnitude_db < best.magnitude_db ? point : best, points[0]);
  const minimumIndex = points.indexOf(minimum);
  const gamma = Math.hypot(minimum.real, minimum.imag);
  const vswr = gamma < 1 ? (1 + gamma) / (1 - gamma) : Number.POSITIVE_INFINITY;
  let bandStart = minimumIndex;
  let bandStop = minimumIndex;
  if (minimum.magnitude_db <= -10) {
    while (bandStart > 0 && points[bandStart - 1].magnitude_db <= -10) bandStart -= 1;
    while (bandStop < points.length - 1 && points[bandStop + 1].magnitude_db <= -10) bandStop += 1;
  }
  const sampledBandPoints = minimum.magnitude_db <= -10 ? bandStop - bandStart + 1 : 0;
  const sampledBandwidth = sampledBandPoints >= 2
    ? points[bandStop].frequency_hz - points[bandStart].frequency_hz
    : null;
  const frequencySteps = points.slice(1).map((point, index) => point.frequency_hz - points[index].frequency_hz);
  const minimumStep = frequencySteps.length ? Math.min(...frequencySteps) : null;
  const maximumStep = frequencySteps.length ? Math.max(...frequencySteps) : null;
  const stepLabel = minimumStep === null || maximumStep === null
    ? "single sample"
    : Math.abs(maximumStep - minimumStep) <= Math.max(1, Math.abs(minimumStep) * 1e-9)
      ? `${(minimumStep / 1e6).toFixed(1)} MHz step`
      : `${(minimumStep / 1e6).toFixed(1)}–${(maximumStep / 1e6).toFixed(1)} MHz variable step`;
  const metadata = revision?.spec.metadata && typeof revision.spec.metadata === "object" && !Array.isArray(revision.spec.metadata)
    ? revision.spec.metadata as Record<string, unknown>
    : {};
  const packages = artifact.runtime.packages ?? {};

  const evidence = [
    { label: "Network response", detail: `${points.length} complex S11 samples`, state: "available" },
    { label: "Field snapshot", detail: hasFieldSlice ? "immutable plane captured" : "not requested", state: hasFieldSlice ? "available" : "neutral" },
    { label: "Reference impedance", detail: `${artifact.reference_impedance_ohm.toFixed(1)} Ω`, state: "available" },
    { label: "Convergence trace", detail: "not captured", state: "missing" },
    { label: "Mesh statistics", detail: "not captured", state: "missing" },
    { label: "Port diagnostics", detail: "not captured", state: "missing" },
  ];

  return (
    <section className="panel engineering-evidence" aria-labelledby="rf-evidence-heading">
      <header className="panel-heading">
        <div><p className="eyebrow">Engineering interpretation</p><h2 id="rf-evidence-heading">RF evidence summary</h2></div>
        <span className="schema-pill">immutable run</span>
      </header>
      <div className="evidence-kpis">
        <div><span>Minimum S11</span><strong>{minimum.magnitude_db.toFixed(2)} dB</strong><small>{formatFrequency(minimum.frequency_hz)}</small></div>
        <div><span>VSWR at minimum</span><strong>{Number.isFinite(vswr) ? vswr.toFixed(2) : "∞"}</strong><small>{artifact.reference_impedance_ohm.toFixed(1)} Ω reference</small></div>
        <div><span>−10 dB bandwidth</span><strong>{sampledBandwidth === null ? "Not resolved" : `${(sampledBandwidth / 1e6).toFixed(1)} MHz`}</strong><small>{sampledBandPoints ? `${sampledBandPoints} contiguous samples` : "threshold not reached"}</small></div>
        <div><span>Sweep coverage</span><strong>{formatFrequency(points[0].frequency_hz)} – {formatFrequency(points[points.length - 1].frequency_hz)}</strong><small>{stepLabel}</small></div>
        <div><span>Run duration</span><strong>{duration(run)}</strong><small>queue through artifact</small></div>
      </div>
      <div className="evidence-columns">
        <div>
          <h3>Evidence coverage</h3>
          <div className="evidence-list">
            {evidence.map((item) => (
              <div key={item.label} className={item.state}><span /><strong>{item.label}</strong><small>{item.detail}</small></div>
            ))}
          </div>
        </div>
        <div>
          <h3>Reproducibility</h3>
          <div className="provenance-list">
            <div><span>Run / revision</span><code>{run.id.slice(0, 10)} / {shortHash(run.revision_id)}</code></div>
            <div><span>Spec SHA</span><code>{shortHash(artifact.spec_sha256)}</code></div>
            <div><span>Compiled SHA</span><code>{shortHash(artifact.compiled_sha256)}</code></div>
            <div><span>Runtime</span><code>{artifact.runtime.backend} · Python {artifact.runtime.python_version ?? "unknown"}</code></div>
            <div><span>Packages</span><code>rfx {packages["rfx-fdtd"] ?? "unknown"} · JAX {packages.jax ?? "unknown"}</code></div>
            <div><span>Artifact SHA</span><code>{shortHash(run.artifact_sha256)}</code></div>
          </div>
        </div>
      </div>
      <p className="evidence-boundary"><strong>{String(metadata.fidelity ?? "Unspecified fidelity")}</strong> · {String(metadata.claims ?? "No quantitative claim declared")}. Missing convergence, mesh, and port diagnostics are shown explicitly and are not inferred from a successful lifecycle.</p>
    </section>
  );
}
