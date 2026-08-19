import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LedgerStrip } from "../components/LedgerStrip";
import { Plate } from "../components/Plate";
import { WebCanvas } from "../components/WebCanvas";
import { ApiFailure, api } from "../lib/api";
import { count, energyShort, millis, share } from "../lib/format";
import type {
  AtomHit,
  ComparisonRowDto,
  IndexDetail,
  InspectRequest,
  InspectResponse,
} from "../lib/types";

const DEFAULTS: Omit<InspectRequest, "index"> = {
  query: { mode: "atom", node: null, text: null, model: "", device: "cpu" },
  profile: null,
  // Mirrors `PropagationDto` in server/schemas.py, which reads the measured
  // winner from the library. TypeScript cannot import a Python dataclass, so
  // this copy is manual and drifts silently - max_hop sat at 6 here for a
  // while after the library moved to 8. Change both together, or teach the
  // server to ship its defaults over the wire.
  propagation: {
    seed_energy: 10,
    damping: 0.6,
    threshold_ratio: 0.01,
    max_hop: 8,
    max_nodes: 512,
    split_alpha: 3,
    mass_enabled: false,
  },
  weights: {
    semantic: 0.5,
    entity: 1,
    structural: 0.3,
    derivation: 1,
    learned: 0,
  },
  ablations: {
    dedup: true,
    dedup_sigma: 2,
    dedup_floor: 0.95,
    dedup_include_seeds: true,
    conflict: true,
    polarity: true,
  },
  view: {
    max_nodes: 220,
    max_edges: 900,
    edge_mode: "contributors",
    label_top_n: 12,
  },
  seed_width: 5,
  contact_overfetch: 3,
  k: 5,
  sample_size: 1000,
  sample_seed: 42,
};

export function InspectView({
  index,
  detail,
  runActive,
}: {
  index: string;
  detail: IndexDetail | null;
  runActive: boolean;
}) {
  const [request, setRequest] = useState<InspectRequest>({
    ...DEFAULTS,
    index,
  });
  const [atoms, setAtoms] = useState<AtomHit[]>([]);
  const [search, setSearch] = useState("");
  const [result, setResult] = useState<InspectResponse | null>(null);
  const [error, setError] = useState<{
    message: string;
    hint: string | null;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [layout, setLayout] = useState<"force" | "hops">("force");
  const [selected, setSelected] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    setRequest((current) => ({ ...current, index }));
    setResult(null);
    setSelected(null);
  }, [index]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api
        .atoms(index, search, controller.signal)
        .then(setAtoms)
        .catch(() => undefined);
    }, 220);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [index, search]);

  const run = useCallback(
    (override?: Partial<InspectRequest>) => {
      const body = { ...request, ...override, index };
      if (body.query.mode === "atom" && !body.query.node) return;
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      setBusy(true);
      setError(null);
      api
        .inspect(body, controller.signal)
        .then((response) => {
          setResult(response);
          setSelected(null);
        })
        .catch((failure: unknown) => {
          if (controller.signal.aborted) return;
          const problem =
            failure instanceof ApiFailure
              ? { message: failure.message, hint: failure.hint }
              : { message: String(failure), hint: null };
          setError(problem);
          setResult(null);
        })
        .finally(() => setBusy(false));
    },
    [index, request],
  );

  const pickAtom = (node: string) => {
    const next = {
      ...request,
      query: { ...request.query, mode: "atom" as const, node },
    };
    setRequest(next);
    run(next);
  };

  const selectedNode = useMemo(
    () => result?.scene.nodes.find((node) => node.id === selected) ?? null,
    [result, selected],
  );
  const selectedPath = useMemo(
    () => result?.paths.find((path) => path.node === selected) ?? null,
    [result, selected],
  );

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-4">
        <Plate
          figure={1}
          title="Query"
          eyebrow="what enters the box"
          caption={
            result
              ? `${result.seeds.length} first-contact atom(s), ${energyShort(
                  request.propagation.seed_energy,
                )} energy split by cosine`
              : undefined
          }
        >
          <label className="eyebrow block" htmlFor="atom-search">
            corpus atom
          </label>
          <input
            id="atom-search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="search id or title"
            className="mt-1 w-full bg-transparent px-2 py-1.5 font-mono text-xs outline-none"
            style={{ boxShadow: "inset 0 0 0 1px var(--color-navy-600)" }}
          />
          <ul className="mt-2 max-h-56 overflow-y-auto">
            {atoms.slice(0, 40).map((atom) => (
              <li key={atom.id}>
                <button
                  type="button"
                  onClick={() => pickAtom(atom.id)}
                  className="w-full px-2 py-1 text-left text-xs"
                  style={{
                    background:
                      request.query.node === atom.id
                        ? "var(--color-navy-500)"
                        : "transparent",
                  }}
                >
                  <span
                    className="font-mono"
                    style={{ color: "var(--color-energy)" }}
                  >
                    {atom.id}
                  </span>{" "}
                  <span style={{ color: "var(--color-ink-quiet)" }}>
                    {atom.title}
                  </span>
                </button>
              </li>
            ))}
          </ul>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
            <Knob
              label="damping"
              hint="fraction each atom forwards"
              value={request.propagation.damping}
              min={0.01}
              max={0.99}
              step={0.01}
              onChange={(value) =>
                setRequest({
                  ...request,
                  propagation: { ...request.propagation, damping: value },
                })
              }
            />
            <Knob
              label="threshold_ratio"
              hint={`floor = ${(
                request.propagation.threshold_ratio *
                request.propagation.seed_energy
              ).toFixed(3)} energy`}
              value={request.propagation.threshold_ratio}
              min={0}
              max={0.5}
              step={0.01}
              onChange={(value) =>
                setRequest({
                  ...request,
                  propagation: {
                    ...request.propagation,
                    threshold_ratio: value,
                  },
                })
              }
            />
            <Knob
              label="max_hop"
              hint="overflow guard, not the stop rule"
              value={request.propagation.max_hop}
              min={0}
              max={12}
              step={1}
              onChange={(value) =>
                setRequest({
                  ...request,
                  propagation: { ...request.propagation, max_hop: value },
                })
              }
            />
            <Knob
              label="seed_width"
              hint="first-contact atoms"
              value={request.seed_width}
              min={1}
              max={20}
              step={1}
              onChange={(value) =>
                setRequest({ ...request, seed_width: value })
              }
            />
          </div>

          <fieldset className="mt-4">
            <legend className="eyebrow">mechanisms</legend>
            <div className="grid gap-x-6 sm:grid-cols-2 lg:grid-cols-1">
              <Toggle
                label="redundancy → vote (dedup)"
                checked={request.ablations.dedup}
                onChange={(checked) =>
                  setRequest({
                    ...request,
                    ablations: { ...request.ablations, dedup: checked },
                  })
                }
              />
              <Toggle
                label="contradiction"
                checked={request.ablations.conflict}
                disabled={!detail?.nli_edges}
                hint={
                  detail?.nli_edges
                    ? undefined
                    : "no edges_nli.json in this index"
                }
                onChange={(checked) =>
                  setRequest({
                    ...request,
                    ablations: { ...request.ablations, conflict: checked },
                  })
                }
              />
              <Toggle
                label="negative-knowledge atoms"
                checked={request.ablations.polarity}
                onChange={(checked) =>
                  setRequest({
                    ...request,
                    ablations: { ...request.ablations, polarity: checked },
                  })
                }
              />
              <Toggle
                label="node mass"
                checked={request.propagation.mass_enabled}
                onChange={(checked) =>
                  setRequest({
                    ...request,
                    propagation: {
                      ...request.propagation,
                      mass_enabled: checked,
                    },
                  })
                }
              />
            </div>
          </fieldset>

          <fieldset className="mt-4">
            <legend className="eyebrow">edge layers</legend>
            <div className="grid gap-x-6 sm:grid-cols-2 lg:grid-cols-1">
              {(
                [
                  "semantic",
                  "entity",
                  "structural",
                  "derivation",
                  "learned",
                ] as const
              ).map((layer) => {
                // Named `edges`, not `count`: `count` is the pinned en-US
                // formatter, and a bare `toLocaleString()` here rendered
                // "134.771" on a Turkish machine, which reads as a decimal.
                const edges =
                  detail?.layers.find((row) => row.layer === layer)?.edges ?? 0;
                return (
                  <Knob
                    key={layer}
                    label={layer}
                    hint={
                      edges ? `${count(edges)} edges` : "empty in this index"
                    }
                    disabled={edges === 0}
                    value={request.weights[layer]}
                    min={0}
                    max={2}
                    step={0.05}
                    onChange={(value) =>
                      setRequest({
                        ...request,
                        weights: { ...request.weights, [layer]: value },
                      })
                    }
                  />
                );
              })}
            </div>
          </fieldset>

          <button
            type="button"
            onClick={() => run()}
            disabled={busy || runActive}
            className="mt-4 w-full px-3 py-2 font-display tracking-widest uppercase"
            style={{
              background: busy
                ? "var(--color-navy-500)"
                : "var(--color-energy)",
              color: busy ? "var(--color-ink)" : "var(--color-navy-900)",
              opacity: runActive ? 0.4 : 1,
            }}
          >
            {busy ? "spreading…" : "inject the query"}
          </button>
          {runActive ? (
            <p
              className="mt-2 text-xs"
              style={{ color: "var(--color-ink-quiet)" }}
            >
              A measurement run is using this machine; inspection is paused so
              it does not take RAM and CPU from it.
            </p>
          ) : null}
        </Plate>
      </div>

      <div className="min-w-0 space-y-4">
        {error ? (
          <div
            className="plate"
            style={{ boxShadow: "inset 0 0 0 1px var(--color-destroyed)" }}
          >
            <p style={{ color: "var(--color-destroyed-ink)" }}>
              {error.message}
            </p>
            {error.hint ? (
              <pre
                className="mt-2 text-xs"
                style={{ color: "var(--color-ink-quiet)" }}
              >
                {error.hint}
              </pre>
            ) : null}
          </div>
        ) : null}

        {!result && !error ? (
          <Plate
            figure={2}
            title="The box, before the query"
            eyebrow="what is loaded"
            caption={
              detail
                ? `${detail.name}: ${count(detail.nodes)} atoms over ${count(
                    detail.corpus_chunks ?? 0,
                  )} passages`
                : "no index loaded"
            }
          >
            <p style={{ color: "var(--color-ink-quiet)" }}>
              Pick a corpus atom on the left and inject it. The web spreads from
              its first contacts and stops when the energy falls under the
              threshold — there is no result count to set.
            </p>
            {detail ? (
              <div className="mt-4 grid gap-6 sm:grid-cols-2">
                <div>
                  <p className="eyebrow">edge layers</p>
                  <table className="mt-1 w-full text-xs">
                    <tbody className="font-mono">
                      {detail.layers.map((layer) => (
                        <tr key={layer.layer}>
                          <td
                            style={{
                              color: layer.present
                                ? "var(--color-ink)"
                                : "var(--color-ink-quiet)",
                            }}
                          >
                            {layer.layer}
                          </td>
                          <td className="text-right tabular">
                            {layer.present ? count(layer.edges) : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <p className="eyebrow">receipt</p>
                  <dl className="mt-1 grid grid-cols-2 gap-x-3 text-xs">
                    <dt style={{ color: "var(--color-ink-quiet)" }}>
                      questions
                    </dt>
                    <dd className="font-mono tabular">
                      {count(detail.questions ?? 0)}
                    </dd>
                    <dt style={{ color: "var(--color-ink-quiet)" }}>
                      propositions
                    </dt>
                    <dd className="font-mono tabular">
                      {detail.propositions
                        ? count(detail.propositions)
                        : "none"}
                    </dd>
                    <dt style={{ color: "var(--color-ink-quiet)" }}>
                      nli edges
                    </dt>
                    <dd className="font-mono tabular">
                      {detail.nli_edges ? count(detail.nli_edges) : "none"}
                    </dd>
                    <dt style={{ color: "var(--color-ink-quiet)" }}>llm</dt>
                    <dd className="font-mono">{detail.llm_model ?? "—"}</dd>
                    <dt style={{ color: "var(--color-ink-quiet)" }}>results</dt>
                    <dd className="font-mono">
                      {detail.has_results ? "measured" : "not yet"}
                    </dd>
                  </dl>
                </div>
              </div>
            ) : null}
          </Plate>
        ) : null}

        {result ? (
          <>
            <Plate
              figure={2}
              title="Energy ledger"
              eyebrow="what the injection became"
              caption={
                result.ledger.balanced
                  ? `energy ledger, balanced within ${result.ledger.tolerance.toExponential(
                      0,
                    )}`
                  : `energy ledger, ${share(result.ledger.residual_share)} unaccounted`
              }
            >
              <LedgerStrip ledger={result.ledger} />
            </Plate>

            <Plate
              figure={3}
              title="Run"
              eyebrow="the numbers the caller decides on"
              caption={`stop reason "${result.stats.stop_reason}" at hop ${result.stats.hop_depth}, floor ${result.stats.threshold.toFixed(3)}`}
            >
              <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
                <Stat
                  label="atoms lit"
                  value={String(result.stats.node_count)}
                  note="activated set size"
                />
                <Stat
                  label="hop depth"
                  value={String(result.stats.hop_depth)}
                  note="how far it reached"
                />
                <Stat
                  label="stopped by"
                  value={result.stats.stop_reason}
                  note="threshold is the real rule"
                />
                <Stat
                  label="dedup τ"
                  value={
                    result.ledger.contact_tau === null
                      ? "—"
                      : result.ledger.contact_tau.toFixed(4)
                  }
                  note={
                    result.ledger.contact_tau === null
                      ? "dedup off"
                      : "adaptive, from this query"
                  }
                />
                <Stat
                  label="held energy"
                  value={energyShort(result.ledger.held)}
                  note="the conserved number"
                />
                <Stat
                  label="elapsed"
                  value={millis(result.stats.elapsed_ms)}
                  note={`scene ${millis(result.stats.scene_ms)}`}
                />
              </dl>
            </Plate>

            <Plate
              figure={4}
              title="Activated web vs plain top-k"
              eyebrow={`both cut at k=${result.comparison.web.length}`}
              caption={`${result.comparison.only_in_web.length} passage(s) only the web returned, ${result.comparison.overlap.length} shared`}
            >
              <div className="grid gap-4 lg:grid-cols-2">
                <Column
                  heading="plain top-k — RIVAL"
                  accent="var(--color-structure-ink)"
                  rows={result.comparison.baseline}
                  scoreLabel="cosine"
                  otherLabel="also in web"
                />
                <Column
                  heading="SPIYWEB activated web"
                  accent="var(--color-energy)"
                  rows={result.comparison.web}
                  scoreLabel="energy"
                  otherLabel="also in top-k"
                />
              </div>
            </Plate>

            <Plate
              figure={5}
              title="The web"
              eyebrow="atom size = accumulated energy"
              caption={result.scene.caption}
              actions={
                <div className="flex items-center gap-1">
                  {(["force", "hops"] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => setLayout(mode)}
                      className="px-2 py-1 font-mono text-2xs uppercase"
                      style={{
                        background:
                          layout === mode
                            ? "var(--color-navy-500)"
                            : "transparent",
                        boxShadow: "inset 0 0 0 1px var(--color-rule-hair)",
                      }}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              }
            >
              <WebCanvas
                scene={result.scene}
                layout={layout}
                animate
                selected={selected}
                onSelect={setSelected}
              />
              <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs">
                {result.scene.layer_order.map((layer) => (
                  <li key={layer} className="flex items-center gap-1.5">
                    <span
                      aria-hidden="true"
                      className="inline-block h-0.5 w-5"
                      style={{
                        background:
                          result.scene.legend[layer] ??
                          "var(--color-structure)",
                      }}
                    />
                    <span style={{ color: "var(--color-ink-quiet)" }}>
                      {layer}
                    </span>
                  </li>
                ))}
                <li className="flex items-center gap-1.5">
                  <span
                    aria-hidden="true"
                    className="inline-block h-0 w-5"
                    style={{ borderTop: "2px dashed var(--color-ink-quiet)" }}
                  />
                  <span style={{ color: "var(--color-ink)" }}>
                    dashed = redundancy link cut by dedup
                  </span>
                </li>
              </ul>
              {selectedNode ? (
                <div
                  className="mt-3 border-t pt-3 text-xs"
                  style={{ borderColor: "var(--color-rule-hair)" }}
                >
                  <p
                    className="font-mono"
                    style={{ color: "var(--color-energy)" }}
                  >
                    {selectedNode.id}
                  </p>
                  <p className="mt-1">{selectedNode.title}</p>
                  <p
                    className="mt-1"
                    style={{ color: "var(--color-ink-quiet)" }}
                  >
                    energy {selectedNode.energy.toFixed(3)} · hop{" "}
                    {selectedNode.hop} · {selectedNode.votes} vote(s) · source{" "}
                    {selectedNode.source_id}
                  </p>
                  {selectedPath ? (
                    <p
                      className="mt-1 font-mono"
                      style={{ color: "var(--color-ink-quiet)" }}
                    >
                      {selectedPath.rendered}
                      {selectedPath.converging > 1
                        ? ` (${selectedPath.converging} contributors — converging evidence)`
                        : ""}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </Plate>

            <Plate
              figure={6}
              title="Diagnosis"
              eyebrow="what the run can say about itself"
              caption={`${result.clusters.length} theme cluster(s), ${result.warnings.length} warning(s)`}
            >
              {result.warnings.map((warning) => (
                <p
                  key={warning.message}
                  className="mb-2 border-l-2 pl-2 text-sm"
                  style={{
                    borderColor:
                      warning.kind === "dispute"
                        ? "var(--color-destroyed)"
                        : "var(--color-rule)",
                    color:
                      warning.kind === "dispute"
                        ? "var(--color-destroyed-ink)"
                        : "var(--color-ink)",
                  }}
                >
                  {warning.kind === "dispute" ? "− " : ""}
                  {warning.message}
                </p>
              ))}
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ color: "var(--color-ink-quiet)" }}>
                    <th className="text-left font-normal">top atom</th>
                    <th className="text-right font-normal">atoms</th>
                    <th className="text-right font-normal">energy</th>
                    <th className="text-right font-normal">share</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {result.clusters.map((cluster) => (
                    <tr key={cluster.top_node}>
                      <td>{cluster.top_node}</td>
                      <td className="text-right">{cluster.nodes.length}</td>
                      <td className="text-right">
                        {cluster.energy.toFixed(2)}
                      </td>
                      <td className="text-right">
                        {share(cluster.energy_share)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {result.refusal ? (
                <details className="mt-3 text-xs">
                  <summary
                    className="cursor-pointer"
                    style={{ color: "var(--color-ink-quiet)" }}
                  >
                    structural refusal report (template-built, no LLM)
                  </summary>
                  <pre className="mt-2 whitespace-pre-wrap">
                    {result.refusal.text}
                  </pre>
                </details>
              ) : null}
            </Plate>
          </>
        ) : null}
      </div>
    </div>
  );
}

function Column({
  heading,
  accent,
  rows,
  scoreLabel,
  otherLabel,
}: {
  heading: string;
  accent: string;
  rows: ComparisonRowDto[];
  scoreLabel: string;
  otherLabel: string;
}) {
  return (
    <div>
      <h3
        className="sticky top-0 z-1 pb-1 font-display text-xs tracking-widest"
        style={{ color: accent, background: "var(--color-navy-700)" }}
      >
        {heading}
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr style={{ color: "var(--color-ink-quiet)" }}>
            <th className="w-6 text-left font-normal">#</th>
            <th className="text-left font-normal">passage</th>
            <th className="text-right font-normal">{scoreLabel}</th>
            <th className="text-right font-normal">{otherLabel}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.node_id} className="align-top">
              <td
                className="tabular"
                style={{ color: "var(--color-ink-quiet)" }}
              >
                {row.rank}
              </td>
              <td className="py-1">
                <span className="font-mono" style={{ color: accent }}>
                  {row.node_id}
                </span>
                <span className="ml-2">{row.title}</span>
                {row.badges.length ? (
                  <span
                    className="ml-2 font-mono text-2xs"
                    style={{ color: "var(--color-rule)" }}
                  >
                    {row.badges.join(" · ")}
                  </span>
                ) : null}
              </td>
              <td className="text-right font-mono tabular">
                {row.score.toFixed(3)}
              </td>
              <td className="text-right">
                {row.in_other ? (
                  <span style={{ color: "var(--color-ink-quiet)" }}>
                    shared
                  </span>
                ) : (
                  <span style={{ color: accent }}>only here</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Stat({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="font-display text-xl tabular">{value}</dd>
      <p className="text-2xs" style={{ color: "var(--color-ink-quiet)" }}>
        {note}
      </p>
    </div>
  );
}

function Knob({
  label,
  hint,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <div style={{ opacity: disabled ? 0.45 : 1 }}>
      <div className="flex items-baseline justify-between">
        <label className="eyebrow" htmlFor={`knob-${label}`}>
          {label}
        </label>
        <span className="font-mono text-xs tabular">{value}</span>
      </div>
      <input
        id={`knob-${label}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full"
        style={{ accentColor: "var(--color-energy)" }}
      />
      {hint ? (
        <p className="text-2xs" style={{ color: "var(--color-ink-quiet)" }}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}

function Toggle({
  label,
  checked,
  disabled,
  hint,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  hint?: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div style={{ opacity: disabled ? 0.45 : 1 }}>
      <label className="flex items-center gap-2 py-0.5 text-xs">
        <input
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
          style={{ accentColor: "var(--color-energy)" }}
        />
        {label}
      </label>
      {hint ? (
        <p
          className="pl-6 text-2xs"
          style={{ color: "var(--color-ink-quiet)" }}
        >
          {hint}
        </p>
      ) : null}
    </div>
  );
}
