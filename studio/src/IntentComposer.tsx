interface IntentComposerProps {
  intent: string;
  onIntentChange: (value: string) => void;
  onSubmit: () => void;
  pending: boolean;
  providerLabel: string;
  compact?: boolean;
}

const examples = [
  "2.4 GHz FR4 patch antenna를 CPU smoke 크기로 만들어줘",
  "WR-90 TE10 도파관의 양방향 S-parameter 실험을 만들어줘",
  "유전체 slab의 Fresnel 반사·투과를 비교하고 싶어",
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
        <span className="copilot-spark" aria-hidden="true">✦</span>
        <textarea
          aria-label={compact ? "Ask the design copilot" : "Describe an RF experiment"}
          value={intent}
          onChange={(event) => onIntentChange(event.target.value)}
          placeholder={compact
            ? "결과를 바탕으로 바꾸고 싶은 목표나 제약을 설명하세요…"
            : "예: 2.4 GHz FR4 패치 안테나를 만들고 S11과 Ez field를 보고 싶어"}
          rows={compact ? 1 : 3}
          maxLength={4000}
        />
        <button className="primary intent-submit" type="submit" disabled={pending || !intent.trim()}>
          {pending ? "Designing…" : compact ? "Propose change" : "Generate proposal"}
        </button>
      </div>
      <div className="intent-meta">
        <span>{providerLabel}</span>
        <span>Proposal only · nothing runs or saves automatically</span>
      </div>
      {!compact && (
        <div className="intent-examples" aria-label="Example RF intents">
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
