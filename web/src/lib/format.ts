/**
 * Formatters. None of them round something away that the reader needs.
 *
 * The energy ledger in particular must never present a residual as zero
 * because it looked small; `energy` keeps four decimals and `share` shows
 * three, so a 0.4% discrepancy stays visible.
 */

export const energy = (value: number): string => value.toFixed(4);
export const energyShort = (value: number): string => value.toFixed(2);

export const share = (value: number): string => `${(value * 100).toFixed(1)}%`;
export const sharePrecise = (value: number): string =>
  `${(value * 100).toFixed(3)}%`;

export const count = (value: number): string => value.toLocaleString("en-US");

export const seconds = (value: number): string => {
  if (!Number.isFinite(value) || value < 0) return "—";
  const total = Math.floor(value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${String(secs).padStart(2, "0")}s`;
  return `${secs}s`;
};

export const millis = (value: number): string =>
  value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;

export const bytes = (value: number): string => {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(0)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
};

/** A signed difference, so a regression never reads as an improvement. */
export const signed = (value: number, digits = 4): string =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
