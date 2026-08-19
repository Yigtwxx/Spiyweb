/**
 * A stacked bar with a ruler under it.
 *
 * The ruler is measurement furniture: ticks, the zero line, the scale. It
 * never carries a value. Every coloured segment is data.
 *
 * `marker` draws a limit line across the bar - a budget, not a reading. It is
 * the one red mark on an otherwise blue gauge, so "red" keeps meaning "the
 * line you must not cross" instead of doubling as a fill colour.
 */

export type Slice = {
  key: string;
  label: string;
  value: number;
  color: string;
  /** Cross-hatched fill, so the meaning survives without colour vision. */
  hatched?: boolean;
};

export function Meter({
  slices,
  total,
  height = 26,
  ticks = 5,
  marker,
  ariaLabel,
}: {
  slices: Slice[];
  total: number;
  height?: number;
  ticks?: number;
  /** Limit line as a fraction of `total`, e.g. the 88% VRAM budget. */
  marker?: { at: number; label: string };
  ariaLabel: string;
}) {
  const safeTotal = total > 0 ? total : 1;
  const summary = slices
    .map((slice) => `${slice.label} ${slice.value.toFixed(4)}`)
    .join(", ");
  return (
    <div>
      <div
        className="relative flex w-full overflow-hidden"
        style={{ height }}
        role="img"
        aria-label={`${ariaLabel}: ${summary}${
          marker ? `, ${marker.label}` : ""
        }`}
      >
        {slices.map((slice) => {
          const width = Math.max(0, (slice.value / safeTotal) * 100);
          if (width <= 0) return null;
          return (
            <div
              key={slice.key}
              title={`${slice.label} — ${slice.value.toFixed(4)}`}
              style={{
                width: `${width}%`,
                background: slice.hatched
                  ? `repeating-linear-gradient(45deg, ${slice.color} 0 4px, transparent 4px 8px)`
                  : slice.color,
                boxShadow: "inset 0 0 0 1px rgb(7 16 33 / 0.55)",
              }}
            />
          );
        })}
        {marker ? (
          <span
            title={marker.label}
            className="pointer-events-none absolute top-0 bottom-0"
            style={{
              left: `${Math.min(100, Math.max(0, marker.at * 100))}%`,
              width: 2,
              background: "var(--color-energy)",
            }}
          />
        ) : null}
      </div>
      <svg
        className="w-full"
        height={12}
        viewBox="0 0 100 12"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <line
          x1="0"
          y1="0.5"
          x2="100"
          y2="0.5"
          stroke="var(--color-rule-hair)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        {Array.from({ length: ticks + 1 }, (_, index) => {
          const x = (index / ticks) * 100;
          const major = index === 0 || index === ticks;
          return (
            <line
              key={index}
              x1={x}
              y1="0"
              x2={x}
              y2={major ? 8 : 4}
              stroke={major ? "var(--color-rule)" : "var(--color-rule-hair)"}
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>
    </div>
  );
}
