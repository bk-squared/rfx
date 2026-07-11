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
        aria-label="Design copilot proposal"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div><p className="eyebrow">Intent to RF evidence</p><h2>Design copilot</h2></div>
          <button aria-label="Close design copilot" onClick={onClose}>×</button>
        </header>
        <div className="agent-security-note copilot-safety">
          <span>◇</span>
          <p><strong>Proposal sandbox</strong>The model can return only a bounded ExperimentSpec patch. Python stays deterministic and no revision or run is created here.</p>
        </div>

        {pending && (
          <div className="copilot-loading" role="status">
            <span>✦</span><strong>Compiling the proposal…</strong>
            <p>Checking schema, geometry, preflight, generated Python, and CPU budget.</p>
          </div>
        )}
        {error && <p className="error-banner copilot-error" role="alert">{error}</p>}
        {!proposal && !pending && !error && (
          <div className="copilot-loading copilot-idle">
            <span>✦</span><strong>Start with an engineering intent</strong>
            <p>Close this panel and describe a new experiment or a change in the prompt bar. A selected successful run is cited automatically.</p>
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
                <div className="clarification"><strong>One decision needed</strong>{proposal.question}</div>
              )}
            </section>

            <section aria-labelledby="proposal-impact-heading">
              <div className="agent-section-title">
                <h3 id="proposal-impact-heading">Review gates</h3>
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
                <h3 id="proposal-diff-heading">Semantic patch</h3>
                <span>not applied</span>
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
              ) : <p className="muted copilot-empty">No patch until the clarification is answered.</p>}
            </section>

            <section aria-labelledby="proposal-reason-heading">
              <div className="agent-section-title"><h3 id="proposal-reason-heading">Why this change</h3></div>
              <ul className="copilot-list">
                {proposal.rationale.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </section>

            <section aria-labelledby="proposal-effect-heading">
              <div className="agent-section-title"><h3 id="proposal-effect-heading">Expected, not yet proven</h3></div>
              <ul className="copilot-list expected">
                {proposal.expected_effects.map((item) => <li key={item}>{item}</li>)}
              </ul>
              <ul className="copilot-list caveats">
                {proposal.caveats.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </section>

            <div className="copilot-actions">
              <p>{proposal.base_revision_id
                ? "Loading keeps this as an uncommitted draft. Review the live geometry/code diff, then approve a revision."
                : "Creating the workspace is the first durable state change and uses this exact compiled spec."}</p>
              <button
                className="primary"
                onClick={onUseProposal}
                disabled={proposal.needs_clarification || !preflightReady}
              >
                {proposal.base_revision_id ? "Load as uncommitted draft" : "Approve & create workspace"}
              </button>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
