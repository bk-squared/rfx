import type { DesignProposal } from "./types";

interface CopilotPanelProps {
  proposal: DesignProposal | null;
  pending: boolean;
  error: string;
  onClose: () => void;
  onUseProposal: () => void;
}

export function CopilotPanel({
  proposal,
  pending,
  error,
  onClose,
  onUseProposal,
}: CopilotPanelProps) {
  const preflightReady = Boolean(proposal?.preview.preflight.ok);
  return (
    <div className="agent-scrim" role="presentation" onMouseDown={onClose}>
      <aside
        className="agent-panel copilot-panel"
        aria-label="Design change review"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div><p className="eyebrow">Draft a model change</p><h2>Design assistant</h2></div>
          <button aria-label="Close design assistant" onClick={onClose}>×</button>
        </header>
        <div className="agent-security-note copilot-safety">
          <span>◇</span>
          <p><strong>Draft only</strong>The assistant can edit ExperimentSpec fields only. No revision is saved and no solver run starts from this panel.</p>
        </div>

        {pending && (
          <div className="copilot-loading" role="status">
            <span>Δ</span><strong>Checking proposed setup…</strong>
            <p>Validating the model, mesh estimate, preflight, generated Python, and CPU cost.</p>
          </div>
        )}
        {error && <p className="error-banner copilot-error" role="alert">{error}</p>}
        {!proposal && !pending && !error && (
          <div className="copilot-loading copilot-idle">
            <span>Δ</span><strong>No design change loaded</strong>
            <p>Describe a parameter change or analysis target in the input bar. When a run is selected, its recorded values are available as read-only context.</p>
          </div>
        )}

        {proposal && !pending && (
          <>
            <section className="copilot-answer" aria-labelledby="copilot-summary-heading">
              <div className="proposal-provider">
                <span>{proposal.provider}</span><code>{proposal.model}</code>
              </div>
              <h3 id="copilot-summary-heading">{proposal.summary}</h3>
              <p>{proposal.answer}</p>
              {proposal.needs_clarification && (
                <div className="clarification"><strong>Input required</strong>{proposal.question}</div>
              )}
            </section>

            <section aria-labelledby="proposal-impact-heading">
              <div className="agent-section-title">
                <h3 id="proposal-impact-heading">Preflight & CPU cost</h3>
                <span>{proposal.patch.length} changes</span>
              </div>
              <div className="proposal-gates">
                <div><span>Compiler</span><strong className={preflightReady ? "good" : "bad"}>{preflightReady ? "PASS" : "BLOCK"}</strong></div>
                <div><span>Grid</span><strong>{proposal.cpu_estimate.grid_shape.join(" × ")}</strong></div>
                <div><span>Cells</span><strong>{proposal.cpu_estimate.estimated_cells.toLocaleString()}</strong></div>
                <div><span>Peak memory</span><strong>≈ {proposal.cpu_estimate.estimated_peak_memory_mb} MB</strong></div>
                <div><span>Execution</span><strong>CPU · {proposal.cpu_estimate.n_steps} steps</strong></div>
              </div>
            </section>

            <section aria-labelledby="proposal-diff-heading">
              <div className="agent-section-title">
                <h3 id="proposal-diff-heading">Proposed changes</h3>
                <span>unsaved</span>
              </div>
              {proposal.patch.length ? (
                <ol className="copilot-diff">
                  {proposal.patch.map((operation, index) => (
                    <li key={`${operation.op}-${operation.path}-${index}`}>
                      <code>{operation.op}</code>
                      <strong>{operation.path}</strong>
                      <span>{operation.op === "remove" ? "removed" : JSON.stringify(operation.value)}</span>
                    </li>
                  ))}
                </ol>
              ) : <p className="muted copilot-empty">No changes until the required input is provided.</p>}
            </section>

            <section aria-labelledby="proposal-reason-heading">
              <div className="agent-section-title"><h3 id="proposal-reason-heading">Engineering basis</h3></div>
              <ul className="copilot-list">
                {proposal.rationale.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </section>

            <section aria-labelledby="proposal-effect-heading">
              <div className="agent-section-title"><h3 id="proposal-effect-heading">Expected effect & limitations</h3></div>
              <ul className="copilot-list expected">
                {proposal.expected_effects.map((item) => <li key={item}>{item}</li>)}
              </ul>
              <ul className="copilot-list caveats">
                {proposal.caveats.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </section>

            <div className="copilot-actions">
              <p>{proposal.base_revision_id
                ? "Load the draft, review geometry, setup, and code, then save a new revision when ready."
                : "Review the compiled setup and preflight before creating the first saved revision."}</p>
              <button
                className="primary"
                onClick={onUseProposal}
                disabled={proposal.needs_clarification || !preflightReady}
              >
                {proposal.base_revision_id ? "Load draft for review" : "Create study from reviewed draft"}
              </button>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
