import type { S11Artifact, S11Point } from "./types";

const bounds = { width: 660, height: 220, left: 52, right: 18, top: 16, bottom: 34 };

const linePath = (points: S11Point[]) => {
  if (!points.length) return "";
  const minX = Math.min(...points.map((point) => point.frequency_hz));
  const maxX = Math.max(...points.map((point) => point.frequency_hz));
  const minY = Math.min(-40, ...points.map((point) => point.magnitude_db));
  const maxY = Math.max(0, ...points.map((point) => point.magnitude_db));
  return points
    .map((point, index) => {
      const x = bounds.left + ((point.frequency_hz - minX) / Math.max(maxX - minX, 1)) * (bounds.width - bounds.left - bounds.right);
      const y = bounds.top + ((maxY - point.magnitude_db) / Math.max(maxY - minY, 1)) * (bounds.height - bounds.top - bounds.bottom);
      return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
};

export function S11Plot({ artifact }: { artifact: S11Artifact }) {
  const minimum = artifact.points.reduce((best, point) =>
    point.magnitude_db < best.magnitude_db ? point : best,
  );
  return (
    <section className="result-card" aria-labelledby="s11-heading">
      <header className="card-header">
        <div>
          <p className="eyebrow">One-port response</p>
          <h3 id="s11-heading">S11 magnitude</h3>
        </div>
        <div className="metric">
          <strong>{minimum.magnitude_db.toFixed(1)} dB</strong>
          <span>@ {(minimum.frequency_hz / 1e9).toFixed(2)} GHz</span>
        </div>
      </header>
      <svg viewBox={`0 0 ${bounds.width} ${bounds.height}`} role="img" aria-label="S11 magnitude in decibels over frequency">
        <line x1="52" x2="642" y1="186" y2="186" className="chart-axis" />
        <line x1="52" x2="52" y1="16" y2="186" className="chart-axis" />
        {[0, -10, -20, -30, -40].map((tick, index) => (
          <g key={tick}>
            <line x1="52" x2="642" y1={16 + index * 42.5} y2={16 + index * 42.5} className="chart-grid" />
            <text x="44" y={21 + index * 42.5} textAnchor="end" className="chart-label">{tick}</text>
          </g>
        ))}
        <path d={linePath(artifact.points)} className="chart-line" />
        <text x="347" y="214" textAnchor="middle" className="chart-label">Frequency (GHz)</text>
        <text x="14" y="105" textAnchor="middle" transform="rotate(-90 14 105)" className="chart-label">|S11| (dB)</text>
      </svg>
    </section>
  );
}

export function SmithChart({ artifact }: { artifact: S11Artifact }) {
  const size = 220;
  const center = size / 2;
  const radius = 86;
  const points = artifact.points.map((point) => `${center + point.real * radius},${center - point.imag * radius}`).join(" ");
  return (
    <section className="result-card compact" aria-labelledby="smith-heading">
      <header className="card-header">
        <div><p className="eyebrow">Complex reflection</p><h3 id="smith-heading">Smith chart</h3></div>
      </header>
      <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Smith chart of complex S11 samples">
        <circle cx={center} cy={center} r={radius} className="smith-ring" />
        <circle cx={center + radius / 2} cy={center} r={radius / 2} className="smith-grid" />
        <path d={`M${center - radius},${center} H${center + radius}`} className="smith-grid" />
        <path d={`M${center},${center - radius} A${radius},${radius} 0 0 0 ${center},${center + radius}`} className="smith-grid" />
        <polyline points={points} className="smith-line" />
      </svg>
    </section>
  );
}
