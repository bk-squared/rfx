import type {
  ReflectionTransmissionArtifact,
  Revision,
  RunRecord,
  SParameterValue,
  SParametersArtifact,
} from "./types";

type FieldStatus = "not-requested" | "loading" | "available" | "unavailable";

interface EvidenceProps<T> {
  artifact: T;
  run: RunRecord;
  revision?: Revision;
  fieldStatus: FieldStatus;
}

const chart = { width: 660, height: 236, left: 54, right: 18, top: 18, bottom: 38 };
const magnitude = (value: SParameterValue) => Math.hypot(value.real, value.imag);
const shortHash = (value: string | null | undefined) => value ? value.slice(0, 10) : "not available";
const frequency = (value: number) => `${(value / 1e9).toFixed(3)} GHz`;
const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
const duration = (run: RunRecord) => {
  const terminal = [...run.events].reverse().find((event) =>
    ["run_succeeded", "run_failed", "run_cancelled"].includes(event.type));
  const elapsed = new Date(terminal?.created_at ?? run.updated_at).getTime()
    - new Date(run.created_at).getTime();
  if (!Number.isFinite(elapsed) || elapsed < 0) return "not available";
  return elapsed < 1000 ? `${elapsed} ms` : `${(elapsed / 1000).toFixed(1)} s`;
};

const metadataLabels = (revision?: Revision) => {
  const metadata = revision?.spec.metadata && typeof revision.spec.metadata === "object"
    && !Array.isArray(revision.spec.metadata)
    ? revision.spec.metadata as Record<string, unknown>
    : {};
  return {
    fidelity: String(metadata.fidelity ?? "Unspecified run class") === "structural-cpu-smoke"
      ? "CPU screening run"
      : String(metadata.fidelity ?? "Unspecified run class"),
    claim: String(metadata.claims ?? "No result-use statement") === "not-for-quantitative-rf-validation"
      ? "No quantitative RF claim"
      : String(metadata.claims ?? "No result-use statement"),
  };
};

const fieldDetail = (status: FieldStatus) => ({
  available: "requested plane captured",
  loading: "loading requested output",
  unavailable: "requested output unavailable",
  "not-requested": "not requested",
})[status];

const xyPath = (
  points: Array<{ x: number; y: number }>,
  yMin: number,
  yMax: number,
) => {
  if (!points.length) return "";
  const xMin = Math.min(...points.map((point) => point.x));
  const xMax = Math.max(...points.map((point) => point.x));
  return points.map((point, index) => {
    const x = chart.left + ((point.x - xMin) / Math.max(xMax - xMin, 1))
      * (chart.width - chart.left - chart.right);
    const clamped = Math.max(yMin, Math.min(yMax, point.y));
    const y = chart.top + ((yMax - clamped) / (yMax - yMin))
      * (chart.height - chart.top - chart.bottom);
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
};

function Provenance({
  run,
  specSha,
  compiledSha,
  backend,
}: {
  run: RunRecord;
  specSha: string;
  compiledSha: string;
  backend: string;
}) {
  return (
    <div className="provenance-list">
      <div><span>Run / revision</span><code>{run.id.slice(0, 10)} / {shortHash(run.revision_id)}</code></div>
      <div><span>Spec SHA</span><code>{shortHash(specSha)}</code></div>
      <div><span>Compiled SHA</span><code>{shortHash(compiledSha)}</code></div>
      <div><span>Artifact SHA</span><code>{shortHash(run.artifact_sha256)}</code></div>
      <div><span>Runtime</span><code>{backend} · {duration(run)}</code></div>
    </div>
  );
}

export function SMatrixEvidence({ artifact, run, revision, fieldStatus }: EvidenceProps<SParametersArtifact>) {
  const points = [...artifact.points].sort((left, right) => left.frequency_hz - right.frequency_hz);
  const s11 = points.map((point) => ({ frequency_hz: point.frequency_hz, value: point.matrix[0][0] }));
  const s21 = points.map((point) => ({ frequency_hz: point.frequency_hz, value: point.matrix[1]?.[0] }));
  const s12 = points.map((point) => ({ frequency_hz: point.frequency_hz, value: point.matrix[0]?.[1] }));
  const worstReflection = s11.reduce((worst, point) =>
    point.value.magnitude_db > worst.value.magnitude_db ? point : worst, s11[0]);
  const weakestTransmission = s21.reduce((worst, point) =>
    point.value.magnitude_db < worst.value.magnitude_db ? point : worst, s21[0]);
  const reciprocityDelta = Math.max(...s21.map((point, index) =>
    Math.abs(magnitude(point.value) - magnitude(s12[index].value))));
  const maximumInputPower = Math.max(...points.flatMap((point) =>
    point.matrix[0].map((_, input) =>
      point.matrix.reduce((sum, row) => sum + magnitude(row[input]) ** 2, 0))));
  const labels = metadataLabels(revision);
  return (
    <section className="panel engineering-evidence workflow-evidence" aria-labelledby="s-matrix-summary-heading">
      <header className="panel-heading">
        <div><p className="eyebrow">Solved two-port network</p><h2 id="s-matrix-summary-heading">S-matrix run summary</h2></div>
        <span className="schema-pill">{artifact.port_names.length} ports · {points.length} frequencies</span>
      </header>
      <div className="evidence-kpis">
        <div><span>Worst S11</span><strong>{worstReflection.value.magnitude_db.toFixed(2)} dB</strong><small>{frequency(worstReflection.frequency_hz)}</small></div>
        <div><span>Weakest S21</span><strong>{weakestTransmission.value.magnitude_db.toFixed(2)} dB</strong><small>{frequency(weakestTransmission.frequency_hz)}</small></div>
        <div><span>Reciprocity Δ</span><strong>{reciprocityDelta.toExponential(2)}</strong><small>max ||S21| − |S12||</small></div>
        <div><span>Input power sum</span><strong>{maximumInputPower.toFixed(3)}</strong><small>max sampled column Σ|Sij|²</small></div>
        <div><span>Sweep coverage</span><strong>{frequency(points[0].frequency_hz)} – {frequency(points.at(-1)!.frequency_hz)}</strong><small>{points.length} solved samples</small></div>
      </div>
      {(points.length < 21 || maximumInputPower > 1.05) && (
        <div className="result-advisories">
          {points.length < 21 && <p><strong>Sample density</strong>{points.length} points are suitable for a structural sweep, not a resolved narrow-band network claim.</p>}
          {maximumInputPower > 1.05 && <p><strong>Power witness</strong>The sampled input-column power sum exceeds 1.05; inspect normalization and convergence before interpreting passivity.</p>}
        </div>
      )}
      <div className="evidence-columns">
        <div><h3>Available outputs</h3><div className="evidence-list">
          <div className="available"><span /><strong>Full S-matrix</strong><small>{artifact.port_names.join(" ↔ ")}</small></div>
          <div className={fieldStatus === "unavailable" ? "missing" : fieldStatus === "available" ? "available" : "neutral"}><span /><strong>Field snapshot</strong><small>{fieldDetail(fieldStatus)}</small></div>
          <div className="missing"><span /><strong>Port diagnostics</strong><small>not recorded</small></div>
        </div></div>
        <div><h3>Run provenance</h3><Provenance run={run} specSha={artifact.spec_sha256} compiledSha={artifact.compiled_sha256} backend={artifact.runtime.backend} /></div>
      </div>
      <p className="evidence-boundary"><strong>{labels.fidelity}</strong> · {labels.claim}. Reciprocity and power values are sampled witnesses, not convergence evidence.</p>
    </section>
  );
}

export function SMatrixPlot({ artifact }: { artifact: SParametersArtifact }) {
  const traces = [
    { label: "S11", output: 0, input: 0, className: "s11" },
    { label: "S21", output: 1, input: 0, className: "s21" },
    { label: "S12", output: 0, input: 1, className: "s12" },
    { label: "S22", output: 1, input: 1, className: "s22" },
  ].filter((trace) => artifact.points[0]?.matrix[trace.output]?.[trace.input]);
  return (
    <section className="result-card network-plot" aria-labelledby="s-matrix-plot-heading">
      <header className="card-header"><div><p className="eyebrow">Network magnitude</p><h3 id="s-matrix-plot-heading">Two-port S-parameters</h3></div><div className="chart-legend">{traces.map((trace) => <span className={trace.className} key={trace.label}>{trace.label}</span>)}</div></header>
      <svg viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label="S11 S21 S12 and S22 magnitude in decibels over frequency">
        <line x1={chart.left} x2={chart.width - chart.right} y1={chart.height - chart.bottom} y2={chart.height - chart.bottom} className="chart-axis" />
        <line x1={chart.left} x2={chart.left} y1={chart.top} y2={chart.height - chart.bottom} className="chart-axis" />
        {[0, -15, -30, -45, -60].map((tick, index) => <g key={tick}><line x1={chart.left} x2={chart.width - chart.right} y1={chart.top + index * 45} y2={chart.top + index * 45} className="chart-grid" /><text x={chart.left - 8} y={chart.top + 5 + index * 45} textAnchor="end" className="chart-label">{tick}</text></g>)}
        {traces.map((trace) => <path key={trace.label} d={xyPath(artifact.points.map((point) => ({ x: point.frequency_hz, y: point.matrix[trace.output][trace.input].magnitude_db })), -60, 0)} className={`chart-line ${trace.className}`} />)}
        <text x={chart.width / 2} y={chart.height - 8} textAnchor="middle" className="chart-label">Frequency (GHz)</text>
        <text x="14" y={chart.height / 2} textAnchor="middle" transform={`rotate(-90 14 ${chart.height / 2})`} className="chart-label">Magnitude (dB)</text>
      </svg>
    </section>
  );
}

export function SMatrixSnapshot({ artifact }: { artifact: SParametersArtifact }) {
  const point = artifact.points[Math.floor(artifact.points.length / 2)];
  return (
    <section className="result-card compact matrix-card" aria-labelledby="matrix-snapshot-heading">
      <header className="card-header"><div><p className="eyebrow">Complex network sample</p><h3 id="matrix-snapshot-heading">S-matrix snapshot</h3></div><div className="metric"><strong>{frequency(point.frequency_hz)}</strong><span>middle sweep bin</span></div></header>
      <div className="matrix-grid">
        {point.matrix.flatMap((row, output) => row.map((value, input) => <div key={`${output}-${input}`}><span>S{output + 1}{input + 1}</span><strong>{value.magnitude_db.toFixed(2)} dB</strong><small>{(Math.atan2(value.imag, value.real) * 180 / Math.PI).toFixed(1)}°</small></div>))}
      </div>
    </section>
  );
}

export function FresnelEvidence({ artifact, run, revision, fieldStatus }: EvidenceProps<ReflectionTransmissionArtifact>) {
  const valid = artifact.points.filter((point) => point.signal_valid);
  const sampled = valid.length ? valid : artifact.points;
  const reflectanceError = mean(sampled.map((point) => Math.abs(point.reflection - point.analytic_reflection)));
  const transmittanceError = mean(sampled.map((point) => Math.abs(point.transmission - point.analytic_transmission)));
  const closureError = mean(sampled.map((point) => Math.abs(point.reflection + point.transmission - 1)));
  const peakReflection = sampled.reduce((best, point) => point.reflection > best.reflection ? point : best, sampled[0]);
  const labels = metadataLabels(revision);
  return (
    <section className="panel engineering-evidence workflow-evidence" aria-labelledby="fresnel-summary-heading">
      <header className="panel-heading"><div><p className="eyebrow">Solved slab response</p><h2 id="fresnel-summary-heading">Fresnel run summary</h2></div><span className="schema-pill">FDTD vs transfer matrix</span></header>
      <div className="evidence-kpis">
        <div><span>Mean |ΔR|</span><strong>{reflectanceError.toExponential(2)}</strong><small>sampled vs exact</small></div>
        <div><span>Mean |ΔT|</span><strong>{transmittanceError.toExponential(2)}</strong><small>sampled vs exact</small></div>
        <div><span>Energy closure</span><strong>{closureError.toExponential(2)}</strong><small>mean |R + T − 1|</small></div>
        <div><span>Peak reflection</span><strong>{peakReflection.reflection.toFixed(3)}</strong><small>{frequency(peakReflection.frequency_hz)}</small></div>
        <div><span>Signal witness</span><strong>{valid.length}/{artifact.points.length}</strong><small>bins above floor</small></div>
      </div>
      {!valid.length && <div className="result-advisories"><p><strong>Signal floor</strong>No bin passed the structural signal-floor witness; all plotted metrics are diagnostic only.</p></div>}
      <div className="evidence-columns">
        <div><h3>Available outputs</h3><div className="evidence-list">
          <div className="available"><span /><strong>Reflection / transmission</strong><small>{artifact.points.length} sampled bins</small></div>
          <div className="available"><span /><strong>Transfer-matrix reference</strong><small>exact lossless slab curves</small></div>
          <div className={fieldStatus === "unavailable" ? "missing" : fieldStatus === "available" ? "available" : "neutral"}><span /><strong>Field snapshot</strong><small>{fieldDetail(fieldStatus)}</small></div>
        </div></div>
        <div><h3>Run provenance</h3><Provenance run={run} specSha={artifact.spec_sha256} compiledSha={artifact.compiled_sha256} backend={artifact.runtime.backend} /></div>
      </div>
      <p className="evidence-boundary"><strong>{labels.fidelity}</strong> · {labels.claim}. Error metrics use signal-valid bins when available.</p>
    </section>
  );
}

export function FresnelPlot({ artifact }: { artifact: ReflectionTransmissionArtifact }) {
  const traces = [
    { label: "R FDTD", key: "reflection", className: "reflection" },
    { label: "T FDTD", key: "transmission", className: "transmission" },
    { label: "R exact", key: "analytic_reflection", className: "reflection exact" },
    { label: "T exact", key: "analytic_transmission", className: "transmission exact" },
  ] as const;
  return (
    <section className="result-card fresnel-plot" aria-labelledby="fresnel-plot-heading">
      <header className="card-header"><div><p className="eyebrow">Power coefficients</p><h3 id="fresnel-plot-heading">Reflection & transmission</h3></div><div className="chart-legend">{traces.map((trace) => <span className={trace.className} key={trace.label}>{trace.label}</span>)}</div></header>
      <svg viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label="FDTD and exact Fresnel reflection and transmission over frequency">
        <line x1={chart.left} x2={chart.width - chart.right} y1={chart.height - chart.bottom} y2={chart.height - chart.bottom} className="chart-axis" />
        <line x1={chart.left} x2={chart.left} y1={chart.top} y2={chart.height - chart.bottom} className="chart-axis" />
        {[1, .75, .5, .25, 0].map((tick, index) => <g key={tick}><line x1={chart.left} x2={chart.width - chart.right} y1={chart.top + index * 45} y2={chart.top + index * 45} className="chart-grid" /><text x={chart.left - 8} y={chart.top + 5 + index * 45} textAnchor="end" className="chart-label">{tick.toFixed(2)}</text></g>)}
        {traces.map((trace) => <path key={trace.label} d={xyPath(artifact.points.map((point) => ({ x: point.frequency_hz, y: point[trace.key] })), 0, 1)} className={`chart-line ${trace.className}`} />)}
        <text x={chart.width / 2} y={chart.height - 8} textAnchor="middle" className="chart-label">Frequency (GHz)</text>
        <text x="14" y={chart.height / 2} textAnchor="middle" transform={`rotate(-90 14 ${chart.height / 2})`} className="chart-label">Power coefficient</text>
      </svg>
    </section>
  );
}
