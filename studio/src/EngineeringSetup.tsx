import type { JsonValue, Revision, ScenePreview } from "./types";

type SpecDocument = Record<string, JsonValue>;

interface EngineeringSetupProps {
  spec: SpecDocument;
  scene: ScenePreview;
  preflight: Revision["preflight"];
  onOpenSpec: () => void;
}

const SPEED_OF_LIGHT_M_S = 299_792_458;

const asRecord = (value: JsonValue | undefined): Record<string, JsonValue> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, JsonValue>
    : {};

const asRecords = (value: JsonValue | undefined): Array<Record<string, JsonValue>> =>
  Array.isArray(value)
    ? value.filter((item): item is Record<string, JsonValue> =>
      Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];

const asStrings = (value: JsonValue | undefined): string[] =>
  Array.isArray(value) ? value.map(String) : [];

const finiteNumber = (value: JsonValue | undefined): number | null => {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const formatFrequency = (value: number | null) => {
  if (value === null) return "not set";
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")} GHz`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")} MHz`;
  return `${value.toPrecision(4)} Hz`;
};

const formatLength = (value: number | null) => {
  if (value === null) return "not set";
  if (Math.abs(value) >= 1) return `${value.toFixed(3)} m`;
  if (Math.abs(value) >= 1e-3) return `${(value * 1e3).toFixed(2)} mm`;
  return `${(value * 1e6).toFixed(1)} µm`;
};

const formatInteger = (value: number) => new Intl.NumberFormat("en-US").format(value);

const detailValue = (item: Record<string, JsonValue>, keys: string[]) => {
  for (const key of keys) {
    const value = item[key];
    if (value !== undefined && value !== null) return String(value);
  }
  return "—";
};

export function EngineeringSetup({ spec, scene, preflight, onOpenSpec }: EngineeringSetupProps) {
  const metadata = asRecord(spec.metadata);
  const simulation = asRecord(spec.simulation);
  const boundaries = asRecord(spec.boundaries);
  const execution = asRecord(spec.execution);
  const validation = asRecord(spec.validation);
  const artifacts = asRecord(spec.artifacts);
  const materials = asRecords(spec.materials);
  const geometry = asRecords(spec.geometry);
  const excitations = asRecords(spec.excitations);
  const observations = asRecords(spec.observations);

  const domain = Array.isArray(simulation.domain_m)
    ? simulation.domain_m.map((value) => Number(value)).filter(Number.isFinite)
    : [];
  const cellSize = finiteNumber(simulation.cell_size_m);
  const gridShape = domain.length === 3 && cellSize && cellSize > 0
    ? domain.map((extent) => Math.ceil(extent / cellSize))
    : [];
  const estimatedCells = gridShape.length === 3
    ? gridShape.reduce((product, value) => product * value, 1)
    : 0;
  const freqMax = finiteNumber(simulation.freq_max_hz);
  const cellsPerWavelength = freqMax && cellSize && cellSize > 0
    ? SPEED_OF_LIGHT_M_S / freqMax / cellSize
    : null;
  const geometryFeatureSizes = geometry.flatMap((item) => {
    if (!Array.isArray(item.bounds_m) || item.bounds_m.length !== 2) return [];
    const lower = Array.isArray(item.bounds_m[0]) ? item.bounds_m[0].map(Number) : [];
    const upper = Array.isArray(item.bounds_m[1]) ? item.bounds_m[1].map(Number) : [];
    return lower.length === 3 && upper.length === 3
      ? upper.map((value, index) => Math.abs(value - lower[index])).filter((value) => value > 0)
      : [];
  });
  const excitationFeatureSizes = excitations
    .map((item) => finiteNumber(item.extent_m))
    .filter((value): value is number => value !== null && value > 0);
  const minimumFeature = [...geometryFeatureSizes, ...excitationFeatureSizes].length
    ? Math.min(...geometryFeatureSizes, ...excitationFeatureSizes)
    : null;
  const cellsAcrossMinimumFeature = minimumFeature && cellSize && cellSize > 0
    ? minimumFeature / cellSize
    : null;
  const maximumPermittivity = Math.max(
    1,
    ...materials.map((item) => finiteNumber(item.relative_permittivity) ?? 1),
  );
  const dielectricCellsPerWavelength = cellsPerWavelength
    ? cellsPerWavelength / Math.sqrt(maximumPermittivity)
    : null;

  const materialIds = new Set(materials.map((item) => String(item.id ?? "")));
  const unresolvedMaterials = geometry.filter((item) => {
    const materialId = String(item.material_id ?? "");
    return materialId && materialId !== "pec" && materialId !== "vacuum" && !materialIds.has(materialId);
  });
  const axes = ["x", "y", "z"];
  const boundariesComplete = axes.every((axis) => {
    const pair = asRecord(boundaries[axis]);
    return Boolean(pair.lo && pair.hi);
  });
  const sweepObservation = observations.find((item) =>
    finiteNumber(item.start_hz) !== null && finiteNumber(item.stop_hz) !== null,
  ) ?? excitations.find((item) =>
    finiteNumber(item.start_hz) !== null && finiteNumber(item.stop_hz) !== null,
  );
  const sweepStart = finiteNumber(sweepObservation?.start_hz);
  const sweepStop = finiteNumber(sweepObservation?.stop_hz);
  const sweepPoints = finiteNumber(sweepObservation?.points);
  const fidelity = String(metadata.fidelity ?? "unspecified");
  const claim = String(metadata.claims ?? "unspecified");
  const fidelityLabel = fidelity === "structural-cpu-smoke" ? "CPU screening" : fidelity;
  const claimLabel = claim === "not-for-quantitative-rf-validation"
    ? "No quantitative RF claim"
    : claim;
  const diagnosticOnly = fidelity.includes("smoke") || claim.includes("not-for-quantitative");
  const meshNeedsReview = cellsAcrossMinimumFeature !== null && cellsAcrossMinimumFeature < 4;
  const sweepNeedsReview = sweepPoints !== null && sweepPoints < 21;

  const readiness = [
    {
      label: "Domain & mesh",
      detail: gridShape.length === 3
        ? meshNeedsReview ? `${cellsAcrossMinimumFeature?.toFixed(1)} cells / min feature` : `${gridShape.join("×")} cells`
        : "incomplete",
      state: gridShape.length === 3 ? meshNeedsReview ? "review" : "configured" : "blocked",
    },
    {
      label: "Materials",
      detail: unresolvedMaterials.length ? `${unresolvedMaterials.length} unresolved` : `${materials.length} explicit`,
      state: unresolvedMaterials.length ? "blocked" : "configured",
    },
    {
      label: "Excitations",
      detail: `${excitations.length} configured`,
      state: excitations.length ? "configured" : "blocked",
    },
    {
      label: "Boundaries",
      detail: boundariesComplete ? "all faces" : "incomplete",
      state: boundariesComplete ? "configured" : "blocked",
    },
    {
      label: "Observations",
      detail: sweepNeedsReview ? `${sweepPoints} sweep samples` : `${observations.length} requested`,
      state: observations.length ? sweepNeedsReview ? "review" : "configured" : "blocked",
    },
    {
      label: "Preflight",
      detail: preflight.ok ? "runnable" : `${preflight.n_errors} errors`,
      state: preflight.ok ? "configured" : "blocked",
    },
  ];

  return (
    <div className="setup-workspace">
      <section className="panel setup-summary" aria-labelledby="engineer-setup-heading">
        <header className="panel-heading">
          <div>
            <p className="eyebrow">Model and solver inputs</p>
            <h2 id="engineer-setup-heading">Simulation setup</h2>
          </div>
          <button className="secondary setup-open-spec" onClick={onOpenSpec}>View ExperimentSpec</button>
        </header>
        <div className="setup-status-rail" aria-label="RF setup readiness">
          {readiness.map((item, index) => (
            <div key={item.label} className={`setup-status ${item.state}`}>
              <span>{index + 1}</span>
              <div><strong>{item.label}</strong><small>{item.detail}</small></div>
            </div>
          ))}
        </div>
        <div className={diagnosticOnly ? "engineering-readiness diagnostic" : "engineering-readiness"}>
          <div>
            <span>Result confidence</span>
            <strong>{diagnosticOnly ? "Screening run only" : "Review run class"}</strong>
          </div>
          <p>{diagnosticOnly
            ? "Suitable for workflow checks and a coarse response. Refine the mesh and verify convergence before quantitative RF use."
            : "Check the declared run class and recorded diagnostics before using the results quantitatively."}</p>
        </div>
        {(meshNeedsReview || sweepNeedsReview) && (
          <div className="setup-advisories" aria-label="Engineering setup advisories">
            {meshNeedsReview && <p><strong>Mesh review</strong> The smallest feature is {cellsAcrossMinimumFeature?.toFixed(1)} cells across. Refine to at least 4 cells across before drawing geometry-sensitive conclusions.</p>}
            {sweepNeedsReview && <p><strong>Sweep review</strong> {sweepPoints} points are enough for a coarse sweep, but may miss a narrow resonance or under-resolve bandwidth.</p>}
          </div>
        )}
      </section>

      <div className="setup-grid">
        <section className="panel setup-card" aria-labelledby="model-mesh-heading">
          <header className="panel-heading"><div><p className="eyebrow">Model</p><h3 id="model-mesh-heading">Domain & mesh</h3></div><span className="schema-pill">estimate</span></header>
          <div className="setup-metrics">
            <div><span>Workflow</span><strong>{String(spec.kind ?? scene.workflow)}</strong></div>
            <div><span>Dimensionality</span><strong>{String(simulation.dimensionality ?? "—")}</strong></div>
            <div><span>Domain</span><strong>{domain.length === 3 ? domain.map((value) => formatLength(value)).join(" × ") : "not set"}</strong></div>
            <div><span>Cell size</span><strong>{formatLength(cellSize)}</strong></div>
            <div><span>Grid shape</span><strong>{gridShape.length === 3 ? gridShape.join(" × ") : "not available"}</strong></div>
            <div><span>Grid cells</span><strong>{estimatedCells ? formatInteger(estimatedCells) : "not available"}</strong></div>
            <div><span>Highest frequency</span><strong>{formatFrequency(freqMax)}</strong></div>
            <div><span>Free-space cells / λ</span><strong>{cellsPerWavelength ? cellsPerWavelength.toFixed(1) : "not available"}</strong></div>
            <div><span>Dielectric cells / λ</span><strong>{dielectricCellsPerWavelength ? `${dielectricCellsPerWavelength.toFixed(1)} at εr ${maximumPermittivity.toFixed(2)}` : "not available"}</strong></div>
            <div><span>Minimum feature / cell</span><strong>{minimumFeature && cellsAcrossMinimumFeature ? `${formatLength(minimumFeature)} / ${cellsAcrossMinimumFeature.toFixed(1)} cells` : "not available"}</strong></div>
          </div>
          <p className="setup-footnote">Grid dimensions are estimated from the domain and cell size. Solver mesh statistics are not recorded for this run class.</p>
        </section>

        <section className="panel setup-card" aria-labelledby="materials-heading">
          <header className="panel-heading"><div><p className="eyebrow">Assignments</p><h3 id="materials-heading">Materials & geometry</h3></div><span className="schema-pill">{geometry.length} objects</span></header>
          <div className="setup-list">
            {materials.length ? materials.map((material) => (
              <article key={String(material.id)}>
                <div><strong>{String(material.id)}</strong><small>{String(material.kind ?? "material")}</small></div>
                <dl>
                  <div><dt>εr</dt><dd>{detailValue(material, ["relative_permittivity"])}</dd></div>
                  <div><dt>σ</dt><dd>{detailValue(material, ["conductivity_s_per_m"])} S/m</dd></div>
                </dl>
              </article>
            )) : <p className="setup-empty">No explicit material records; the workflow uses vacuum and boundary-defined conductors.</p>}
          </div>
          <div className="assignment-strip" aria-label="Geometry material assignments">
            {geometry.map((item) => <span key={String(item.id)}><strong>{String(item.id)}</strong> → {String(item.material_id)}</span>)}
            {!geometry.length && <span>Analytic domain; no explicit solids</span>}
          </div>
        </section>

        <section className="panel setup-card" aria-labelledby="excitation-heading">
          <header className="panel-heading"><div><p className="eyebrow">Physics</p><h3 id="excitation-heading">Ports & excitations</h3></div><span className="schema-pill">{excitations.length}</span></header>
          <div className="setup-list">
            {excitations.map((item) => {
              const impedance = finiteNumber(item.impedance_ohm);
              const mode = Array.isArray(item.mode) ? item.mode.join("") : null;
              return (
                <article key={String(item.id)}>
                  <div><strong>{String(item.id)}</strong><small>{String(item.kind)}</small></div>
                  <dl>
                    <div><dt>Direction</dt><dd>{detailValue(item, ["direction"])}</dd></div>
                    <div><dt>Center</dt><dd>{formatFrequency(finiteNumber(item.f0_hz))}</dd></div>
                    <div><dt>Reference</dt><dd>{impedance ? `${impedance.toFixed(1)} Ω` : mode ? `${String(item.mode_type ?? "")} ${mode}` : "workflow-defined"}</dd></div>
                    <div><dt>Plane</dt><dd>{finiteNumber(item.reference_plane_m) !== null
                      ? formatLength(finiteNumber(item.reference_plane_m))
                      : item.kind === "lumped_port" ? "at lumped port" : "not set"}</dd></div>
                  </dl>
                </article>
              );
            })}
          </div>
        </section>

        <section className="panel setup-card" aria-labelledby="boundary-heading">
          <header className="panel-heading"><div><p className="eyebrow">Domain termination</p><h3 id="boundary-heading">Boundary conditions</h3></div><span className="schema-pill">{boundariesComplete ? "complete" : "review"}</span></header>
          <div className="boundary-table">
            {axes.map((axis) => {
              const pair = asRecord(boundaries[axis]);
              return <div key={axis}><strong>{axis.toUpperCase()}</strong><span>lo · {String(pair.lo ?? "—")}</span><span>hi · {String(pair.hi ?? "—")}</span></div>;
            })}
          </div>
          <div className="setup-callout"><span>Absorber</span><strong>{boundaries.cpml_layers ? `${String(boundaries.cpml_layers)} CPML layers` : "No CPML layer count"}</strong></div>
        </section>

        <section className="panel setup-card" aria-labelledby="observation-heading">
          <header className="panel-heading"><div><p className="eyebrow">Study</p><h3 id="observation-heading">Frequency & outputs</h3></div><span className="schema-pill">{observations.length}</span></header>
          <div className="sweep-summary">
            <div><span>Sweep</span><strong>{sweepStart !== null && sweepStop !== null ? `${formatFrequency(sweepStart)} – ${formatFrequency(sweepStop)}` : "workflow-defined"}</strong></div>
            <div><span>Samples</span><strong>{sweepPoints ? `${sweepPoints} points` : "not specified"}</strong></div>
          </div>
          <div className="observation-list">
            {observations.map((item) => (
              <div key={String(item.id)}><span className={String(item.kind).includes("field") ? "field" : "network"} /><strong>{String(item.id)}</strong><small>{String(item.kind)}</small></div>
            ))}
          </div>
        </section>

        <section className="panel setup-card" aria-labelledby="execution-heading">
          <header className="panel-heading"><div><p className="eyebrow">Solver</p><h3 id="execution-heading">Execution & checks</h3></div><span className="schema-pill">CPU</span></header>
          <div className="setup-metrics compact">
            <div><span>Precision</span><strong>{String(simulation.precision ?? "—")}</strong></div>
            <div><span>Time steps</span><strong>{String(execution.n_steps ?? "—")}</strong></div>
            <div><span>S-parameter steps</span><strong>{String(execution.s_param_n_steps ?? "—")}</strong></div>
            <div><span>Timeout</span><strong>{String(execution.timeout_seconds ?? "—")} s</strong></div>
            <div><span>Run class</span><strong>{fidelityLabel}</strong></div>
            <div><span>Result use</span><strong>{claimLabel}</strong></div>
          </div>
          <div className="contract-groups">
            <div><span>Required checks</span><p>{asStrings(validation.required_checks).join(" · ") || "none declared"}</p></div>
            <div><span>Metrics</span><p>{asStrings(validation.metrics).join(" · ") || "none declared"}</p></div>
            <div><span>Saved outputs</span><p>{asStrings(artifacts.save).join(" · ") || "none declared"}</p></div>
          </div>
        </section>
      </div>
    </div>
  );
}
