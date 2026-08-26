import { useCallback, useEffect, useMemo, useState } from "react";

import { LedgerStrip } from "../components/LedgerStrip";
import { Plate } from "../components/Plate";
import { WebCanvas } from "../components/WebCanvas";
import { api, ApiFailure } from "../lib/api";
import { count, energyShort, millis, share } from "../lib/format";
import type {
  Capabilities,
  LedgerDto,
  SceneDto,
  TraceLedgerDto,
  TraceRecordDto,
  TraceSummary,
} from "../lib/types";

/**
 * The trace viewer: what this application's retrieval actually did.
 *
 * The Inspect view next door runs a NEW query and draws it. This one draws
 * calls that already happened — inside somebody's own program, possibly on
 * another machine, possibly last week. That difference is the whole reason
 * the page exists: an explanation of a retrieval nobody made is a demo, and
 * an explanation of the retrieval that actually served a user is evidence.
 *
 * The picture is not drawn here. The server lays the scene out with the same
 * `spiyweb.scene` builder the live inspector uses, so the two views cannot
 * drift into showing different pictures of the same mechanism.
 */

/** The record's flat ledger, in the shape `LedgerStrip` already draws. */
function asLedgerDto(ledger: TraceLedgerDto): LedgerDto {
  const destroyed =
    ledger.destroyed_conflict +
    ledger.destroyed_negative_seed +
    ledger.destroyed_polarity;
  return {
    injected: ledger.injected,
    held: ledger.held,
    dissipated: ledger.dissipated,
    destroyed: {
      conflict: ledger.destroyed_conflict,
      // A record keeps the energy each mechanism destroyed, not how many
      // times it fired — the events are in `record.events`, and inventing
      // counts here to fill the shape would be a lie in a tooltip.
      conflict_events: 0,
      negative_seed: ledger.destroyed_negative_seed,
      negative_seed_events: 0,
      polarity: ledger.destroyed_polarity,
      polarity_events: 0,
      total: destroyed,
    },
    residual: ledger.residual,
    residual_share: ledger.injected > 0 ? ledger.residual / ledger.injected : 0,
    mismatch: ledger.mismatch,
    tolerance: ledger.tolerance,
    balanced: ledger.balanced,
    exact: ledger.exact,
    notes: ledger.notes,
    dedup_cuts: ledger.dedup_cuts,
    contact_cuts: 0,
    dedup_taus: [],
    contact_tau: null,
  };
}

function when(stamp: string): string {
  const parsed = new Date(stamp);
  return Number.isNaN(parsed.valueOf())
    ? stamp
    : parsed.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
}

export function TracesView({ capabilities }: { capabilities: Capabilities }) {
  const [rows, setRows] = useState<TraceSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [record, setRecord] = useState<TraceRecordDto | null>(null);
  const [scene, setScene] = useState<SceneDto | null>(null);
  const [layout, setLayout] = useState<"force" | "hops">("hops");
  const [node, setNode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [question, setQuestion] = useState("");

  const refresh = useCallback(async () => {
    try {
      const page = await api.traces(100, 0);
      setRows(page.traces);
      setTotal(page.total);
      setError(null);
      // Land on the newest call rather than an empty pane: it is the one
      // whoever opened this link is almost certainly here about.
      setSelected((current) =>
        current && page.traces.some((row) => row.trace_id === current)
          ? current
          : (page.traces[0]?.trace_id ?? null),
      );
    } catch (failure) {
      setError(
        failure instanceof ApiFailure ? failure.message : String(failure),
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // A live store grows while the page is open, so it is polled; a file is
  // re-read on demand by the server, and polling it would be busywork.
  useEffect(() => {
    if (capabilities.mode === "file") return;
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [capabilities.mode, refresh]);

  useEffect(() => {
    if (!selected) {
      setRecord(null);
      setScene(null);
      return;
    }
    let live = true;
    setNode(null);
    Promise.all([api.trace(selected), api.traceScene(selected)])
      .then(([full, drawn]) => {
        if (!live) return;
        setRecord(full);
        setScene(drawn);
      })
      .catch((failure) => {
        if (!live) return;
        setError(
          failure instanceof ApiFailure ? failure.message : String(failure),
        );
      });
    return () => {
      live = false;
    };
  }, [selected]);

  const ask = async () => {
    const text = question.trim();
    if (!text) return;
    setAsking(true);
    try {
      const fresh = await api.ask({ query: text });
      await refresh();
      setSelected(fresh.trace_id);
      setQuestion("");
      setError(null);
    } catch (failure) {
      setError(
        failure instanceof ApiFailure ? failure.message : String(failure),
      );
    } finally {
      setAsking(false);
    }
  };

  const passages = useMemo(
    () =>
      record
        ? [...record.nodes]
            .filter((row) => row.energy > 0)
            .sort((a, b) => b.energy - a.energy)
        : [],
    [record],
  );
  const chosen = node
    ? (record?.nodes.find((row) => row.id === node) ?? null)
    : null;
  const chosenPath = node
    ? (record?.paths.find((row) => row.node === node) ?? null)
    : null;

  return (
    <div className="flex flex-col gap-4">
      {error ? (
        <p
          className="font-mono text-xs"
          style={{ color: "var(--color-destroyed-ink)" }}
          role="alert"
        >
          {error}
        </p>
      ) : null}

      {!capabilities.live && capabilities.mode === "store" ? (
        <p
          className="font-plate italic"
          style={{ color: "var(--color-ink-quiet)" }}
        >
          This install holds the index but cannot embed a question, so there
          is no search box — run{" "}
          <code className="font-mono">pip install "spiyweb[embed]"</code> to
          ask it something. Recorded calls still draw.
        </p>
      ) : null}

      {capabilities.live ? (
        <form
          className="flex flex-wrap items-center gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void ask();
          }}
        >
          <label className="eyebrow" htmlFor="ask">
            ask
          </label>
          <input
            id="ask"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="a question for this index"
            className="min-w-[22rem] flex-1 bg-transparent px-2 py-1 font-mono text-xs"
            style={{ boxShadow: "inset 0 0 0 1px var(--color-navy-600)" }}
          />
          <button
            type="submit"
            disabled={asking || !question.trim()}
            className="px-3 py-1 font-display text-xs tracking-widest uppercase disabled:opacity-40"
            style={{
              color: "var(--color-navy-900)",
              background: "var(--color-rule)",
            }}
          >
            {asking ? "asking…" : "ask"}
          </button>
        </form>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <Plate
          figure={1}
          eyebrow={`${capabilities.mode} · ${capabilities.origin}`}
          title="recorded calls"
          caption={`${count(total)} call(s) held${
            capabilities.mode === "store"
              ? " — a ring buffer, so the oldest are gone"
              : ""
          }.`}
        >
          <ol className="flex max-h-[32rem] flex-col overflow-y-auto">
            {rows.map((row) => (
              <li key={row.trace_id}>
                <button
                  onClick={() => setSelected(row.trace_id)}
                  aria-current={row.trace_id === selected}
                  className="w-full border-b px-2 py-2 text-left"
                  style={{
                    borderColor: "var(--color-rule-hair)",
                    background:
                      row.trace_id === selected
                        ? "color-mix(in oklab, var(--color-rule) 14%, transparent)"
                        : "transparent",
                  }}
                >
                  <div className="truncate font-plate text-sm">
                    {row.query || "(no text)"}
                  </div>
                  <div
                    className="flex flex-wrap gap-x-3 font-mono text-[0.7rem]"
                    style={{ color: "var(--color-ink-quiet)" }}
                  >
                    <span>#{row.sequence}</span>
                    <span>{when(row.recorded_at)}</span>
                    <span>{count(row.node_count)} atoms</span>
                    <span>hop {row.hops_used}</span>
                    <span>{millis(row.elapsed_ms)}</span>
                    {row.kind === "colored" ? <span>coloured</span> : null}
                    {row.balanced === false ? (
                      <span style={{ color: "var(--color-destroyed-ink)" }}>
                        ledger off
                      </span>
                    ) : null}
                  </div>
                </button>
              </li>
            ))}
            {rows.length === 0 ? (
              <li
                className="px-2 py-6 font-plate italic"
                style={{ color: "var(--color-ink-quiet)" }}
              >
                Nothing recorded yet. Every query this index answers lands here.
              </li>
            ) : null}
          </ol>
        </Plate>

        <div className="flex flex-col gap-4">
          <Plate
            figure={2}
            eyebrow={record ? `${record.kind} · ${record.stop_reason}` : ""}
            title={record ? record.query || "(no text)" : "the web"}
            caption={scene?.caption}
            actions={
              <div role="group" aria-label="layout" className="flex gap-1">
                {(["hops", "force"] as const).map((name) => (
                  <button
                    key={name}
                    onClick={() => setLayout(name)}
                    aria-pressed={layout === name}
                    className="px-2 py-1 font-display text-[0.7rem] tracking-widest uppercase"
                    style={{
                      color:
                        layout === name
                          ? "var(--color-navy-900)"
                          : "var(--color-ink)",
                      background:
                        layout === name ? "var(--color-rule)" : "transparent",
                      boxShadow:
                        layout === name
                          ? "none"
                          : "inset 0 0 0 1px var(--color-rule-hair)",
                    }}
                  >
                    {name}
                  </button>
                ))}
              </div>
            }
          >
            {scene ? (
              <WebCanvas
                scene={scene}
                layout={layout}
                animate={false}
                selected={node}
                onSelect={setNode}
              />
            ) : (
              <p
                className="px-2 py-8 font-plate italic"
                style={{ color: "var(--color-ink-quiet)" }}
              >
                Pick a recorded call.
              </p>
            )}
          </Plate>

          {record?.ledger ? (
            <Plate
              figure={3}
              title="energy ledger"
              caption={
                record.ledger.balanced
                  ? "held + dissipated + destroyed accounts for every unit injected."
                  : "the reconstruction does not close — that is a finding, not rounding."
              }
            >
              <LedgerStrip ledger={asLedgerDto(record.ledger)} />
            </Plate>
          ) : null}

          <Plate
            figure={4}
            title={chosen ? chosen.id : "what lit up"}
            caption={
              chosen
                ? `${chosen.layer} · ${chosen.source_id} · ${chosen.votes} vote(s)`
                : `${passages.length} activated atom(s), strongest first.`
            }
          >
            {chosen ? (
              <div className="flex flex-col gap-2 px-2 py-2">
                <p className="font-plate">{chosen.text || "(no text kept)"}</p>
                <dl
                  className="grid grid-cols-2 gap-x-6 font-mono text-xs sm:grid-cols-4"
                  style={{ color: "var(--color-ink-quiet)" }}
                >
                  <div>
                    <dt className="eyebrow">energy</dt>
                    <dd>{energyShort(chosen.energy)}</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">hop</dt>
                    <dd>{chosen.hop}</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">votes</dt>
                    <dd>{chosen.votes}</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">seed cosine</dt>
                    <dd>
                      {chosen.seed_similarity === null
                        ? "—"
                        : share(chosen.seed_similarity)}
                    </dd>
                  </div>
                </dl>
                {chosenPath ? (
                  <p className="font-mono text-xs">
                    {chosenPath.steps.join(" → ")}
                    {chosenPath.converging > 1
                      ? ` · ${chosenPath.converging} contributors`
                      : ""}
                  </p>
                ) : null}
              </div>
            ) : (
              <ol className="flex max-h-[24rem] flex-col overflow-y-auto">
                {passages.map((row) => (
                  <li
                    key={row.id}
                    className="border-b px-2 py-2"
                    style={{ borderColor: "var(--color-rule-hair)" }}
                  >
                    <div className="flex items-baseline gap-3">
                      <span className="font-mono text-xs">
                        {energyShort(row.energy)}
                      </span>
                      <span className="font-plate">{row.text || row.id}</span>
                    </div>
                    <div
                      className="font-mono text-[0.7rem]"
                      style={{ color: "var(--color-ink-quiet)" }}
                    >
                      {row.source_id} · hop {row.hop} · {row.votes} vote(s)
                      {row.disputed ? " · disputed" : ""}
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </Plate>
        </div>
      </div>
    </div>
  );
}
