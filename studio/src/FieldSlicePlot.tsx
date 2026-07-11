import type { FieldSliceArtifact } from "./types";

const color = (value: number, maximum: number) => {
  const normalized = maximum > 0 ? Math.max(-1, Math.min(1, value / maximum)) : 0;
  const magnitude = Math.abs(normalized);
  return normalized >= 0
    ? `hsl(38 78% ${12 + magnitude * 58}%)`
    : `hsl(188 72% ${12 + magnitude * 52}%)`;
};

export function FieldSlicePlot({ artifact }: { artifact: FieldSliceArtifact }) {
  const [rows, columns] = artifact.shape;
  const width = 420;
  const height = 250;
  const left = 42;
  const top = 16;
  const plotWidth = width - left - 16;
  const plotHeight = height - top - 36;
  const cellWidth = plotWidth / Math.max(columns, 1);
  const cellHeight = plotHeight / Math.max(rows, 1);
  const component = artifact.component.toUpperCase();

  return (
    <section className="result-card field-card" aria-labelledby="field-slice-heading">
      <header className="card-header">
        <div>
          <p className="eyebrow">Immutable final-state plane</p>
          <h3 id="field-slice-heading">{component} field slice</h3>
        </div>
        <div className="metric">
          <strong>{artifact.maximum_absolute.toExponential(2)}</strong>
          <span>max |{component}| {artifact.units}</span>
        </div>
      </header>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${component} field slice heatmap on the ${artifact.slice_axis} plane`}
        data-testid="field-slice-heatmap"
      >
        <title>
          {component} final field at {artifact.slice_axis}={artifact.actual_coordinate_m} m
        </title>
        <g className="field-heatmap">
          {artifact.values.flatMap((row, rowIndex) =>
            row.map((value, columnIndex) => (
              <rect
                key={`${rowIndex}-${columnIndex}`}
                x={left + columnIndex * cellWidth}
                y={top + (rows - rowIndex - 1) * cellHeight}
                width={cellWidth + 0.25}
                height={cellHeight + 0.25}
                fill={color(value, artifact.maximum_absolute)}
              />
            )),
          )}
        </g>
        <rect x={left} y={top} width={plotWidth} height={plotHeight} className="field-frame" />
        <text x={left + plotWidth / 2} y={height - 8} textAnchor="middle" className="chart-label">
          {artifact.axis_labels[1]} (m)
        </text>
        <text
          x="12"
          y={top + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${top + plotHeight / 2})`}
          className="chart-label"
        >
          {artifact.axis_labels[0]} (m)
        </text>
      </svg>
      <p className="field-caption">
        {artifact.observation_id} · requested {artifact.requested_coordinate_m.toFixed(4)} m · sampled {artifact.actual_coordinate_m.toFixed(4)} m · {rows}×{columns}
      </p>
    </section>
  );
}
