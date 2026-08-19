import { Meter, type Slice } from "./Meter";
import { energy, sharePrecise } from "../lib/format";
import type { LedgerDto } from "../lib/types";

/**
 * The energy ledger — the signature of this page.
 *
 * CLAUDE.md §2.1 claims dedup REDISTRIBUTES energy while contradictions,
 * negative seeds and negative-polarity atoms DESTROY it, and that nothing
 * else creates or destroys any. This strip is that claim, audited, on every
 * query.
 *
 * Two rules it will not break:
 *  - dedup is never a slice. Drawing it as one would teach the reader the
 *    opposite of the invariant.
 *  - a residual is never rounded away. A positive one gets its own hatched
 *    slice; a negative one replaces the chart with an error report, because
 *    "energy appeared from nowhere" is a finding, not a 2% sliver.
 */

export function LedgerStrip({ ledger }: { ledger: LedgerDto }) {
  const broken = ledger.residual < -ledger.tolerance;
  const unaccounted = ledger.residual > ledger.tolerance;

  if (broken) {
    return (
      <div
        className="border-l-2 p-3 text-sm"
        style={{
          borderColor: "var(--color-destroyed)",
          background: "rgb(194 71 127 / 0.08)",
        }}
      >
        <p
          className="font-display tracking-widest uppercase"
          style={{ color: "var(--color-destroyed-ink)" }}
        >
          the ledger does not balance — energy appears to have been created
        </p>
        <p className="mt-2" style={{ color: "var(--color-ink-quiet)" }}>
          This is an error report, not a chart. A negative residual means the
          reconstruction or the core disagrees with itself; drawing it as a tidy
          slice would be a lie.
        </p>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs">
          <Row label="injected" value={energy(ledger.injected)} />
          <Row label="held" value={energy(ledger.held)} />
          <Row label="dissipated" value={energy(ledger.dissipated)} />
          <Row label="destroyed" value={energy(ledger.destroyed.total)} />
          <Row label="residual" value={energy(ledger.residual)} />
          <Row label="mismatch" value={energy(ledger.mismatch)} />
        </dl>
      </div>
    );
  }

  const slices: Slice[] = [
    {
      key: "held",
      label: "held in atoms",
      value: ledger.held,
      color: "var(--color-energy)",
    },
    {
      key: "dissipated",
      label: "forwarded, died under the threshold",
      value: ledger.dissipated,
      color: "var(--color-structure)",
    },
    {
      key: "destroyed",
      label: "destroyed",
      value: ledger.destroyed.total,
      color: "var(--color-destroyed)",
    },
  ];
  if (unaccounted) {
    slices.push({
      key: "residual",
      label: "unaccounted",
      value: ledger.residual,
      color: "var(--color-rule)",
      hatched: true,
    });
  }

  return (
    <div>
      <Meter
        slices={slices}
        total={ledger.injected}
        ariaLabel="energy ledger"
        ticks={5}
      />
      <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs">
        {slices.map((slice) => (
          <li key={slice.key} className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5"
              style={{
                background: slice.hatched
                  ? `repeating-linear-gradient(45deg, ${slice.color} 0 3px, transparent 3px 6px)`
                  : slice.color,
              }}
            />
            <span style={{ color: "var(--color-ink-quiet)" }}>
              {slice.label}
            </span>
            <span className="font-mono tabular">{energy(slice.value)}</span>
          </li>
        ))}
      </ul>

      {ledger.destroyed.total > 0 ? (
        <p
          className="mt-3 text-xs"
          style={{ color: "var(--color-destroyed-ink)" }}
        >
          − destroyed: contradiction {energy(ledger.destroyed.conflict)} (
          {ledger.destroyed.conflict_events} events) · negative seed{" "}
          {energy(ledger.destroyed.negative_seed)} (
          {ledger.destroyed.negative_seed_events}) · polarity{" "}
          {energy(ledger.destroyed.polarity)} (
          {ledger.destroyed.polarity_events})
        </p>
      ) : null}

      {/* Deliberately below the strip, never inside it. */}
      <p
        className="mt-3 border-t pt-2 text-xs"
        style={{
          borderColor: "var(--color-rule-hair)",
          color: "var(--color-ink-quiet)",
        }}
      >
        <strong style={{ color: "var(--color-ink)" }}>
          {ledger.dedup_cuts + ledger.contact_cuts} redundancy link(s) cut
        </strong>{" "}
        — dedup redistributes energy, it never destroys it, so it is not a slice
        above. Adaptive cut τ ={" "}
        <span className="font-mono">
          {ledger.contact_tau === null ? "—" : ledger.contact_tau.toFixed(4)}
        </span>
        {ledger.contact_tau === null
          ? " (dedup off: no similarity backend)"
          : ""}
        {ledger.dedup_taus.length > 1
          ? ` · per-hop: ${ledger.dedup_taus.map((tau) => tau.toFixed(3)).join(", ")}`
          : ""}
      </p>

      {unaccounted ||
      ledger.mismatch > ledger.tolerance ||
      ledger.notes.length ? (
        <details className="mt-2 text-xs">
          <summary
            className="cursor-pointer"
            style={{
              color: unaccounted
                ? "var(--color-rule)"
                : "var(--color-ink-quiet)",
            }}
          >
            {unaccounted
              ? `${sharePrecise(ledger.residual_share)} unaccounted — why?`
              : "reconstruction notes"}
          </summary>
          <div
            className="mt-2 space-y-1"
            style={{ color: "var(--color-ink-quiet)" }}
          >
            <p>
              The core keeps no ledger; this is replayed from the activations
              and the recorded destruction. Residual{" "}
              <span className="font-mono">{energy(ledger.residual)}</span>,
              replay mismatch{" "}
              <span className="font-mono">{energy(ledger.mismatch)}</span>,
              tolerance{" "}
              <span className="font-mono">
                {ledger.tolerance.toExponential(1)}
              </span>
              {ledger.exact ? "" : " (approximate: node mass is on)"}.
            </p>
            {ledger.notes.map((note) => (
              <p key={note}>· {note}</p>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt style={{ color: "var(--color-ink-quiet)" }}>{label}</dt>
      <dd className="tabular">{value}</dd>
    </>
  );
}
