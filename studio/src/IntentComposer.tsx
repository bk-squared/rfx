interface IntentComposerProps {
  intent: string;
  onIntentChange: (value: string) => void;
  onSubmit: () => void;
  pending: boolean;
  providerLabel: string;
  compact?: boolean;
}

const examples = [
  "2.4 GHz FR4 patch · 1.8–3.0 GHz S11 · z=14 mm Ez plane",
  "WR-90 · TE10 · two-port S-parameters",
  "Dielectric slab · Fresnel reflection / transmission",
];

export function IntentComposer({
  intent,
  onIntentChange,
  onSubmit,
  pending,
  providerLabel,
  compact = false,
}: IntentComposerProps) {
  return (
    <form
      className={compact ? "intent-composer compact" : "intent-composer"}
      onSubmit={(event) => {
        event.preventDefault();
        if (intent.trim()) onSubmit();
      }}
    >
      <div className="intent-input-row">
        <span className="copilot-spark" aria-hidden="true">Δ</span>
        <textarea
          aria-label={compact ? "Describe a design change" : "Describe an RF simulation"}
          value={intent}
          onChange={(event) => onIntentChange(event.target.value)}
          placeholder={compact
            ? "예: sweep을 3.5 GHz까지 늘리고 41 points로 변경"
            : "예: 2.4 GHz FR4 patch, 1.8–3.0 GHz S11 sweep, z=14 mm Ez plane"}
          rows={compact ? 1 : 3}
          maxLength={4000}
        />
        <button className="primary intent-submit" type="submit" disabled={pending || !intent.trim()}>
          {pending ? "Checking setup…" : compact ? "Review change" : "Review setup"}
        </button>
      </div>
      <div className="intent-meta">
        <span>{providerLabel}</span>
        <span>Draft only · no revision or run created</span>
      </div>
      {!compact && (
        <div className="intent-examples" aria-label="Example RF studies">
          {examples.map((example) => (
            <button key={example} type="button" onClick={() => onIntentChange(example)}>
              {example}
            </button>
          ))}
        </div>
      )}
    </form>
  );
}
