import { useEffect, useMemo, useState } from "react";
import { Meter } from "../components/Meter";
import { Plate } from "../components/Plate";
import { ApiFailure, api } from "../lib/api";
import { count, seconds, share, signed } from "../lib/format";
import type {
  BootstrapReport,
  IndexSummary,
  ResultsPayload,
  RunPlan,
  RunProgress,
  RunRequest,
  RunStatus,
  SystemStatus,
} from "../lib/types";

const LABEL: Record<string, string> = {
  web: "SPIYWEB",
  topk: "top-k — RIVAL",
  iterative: "iterative — RIVAL",
};

export function RunsView({
  index,
  indexes,
  run,
  system,
}: {
  index: string;
  indexes: IndexSummary[];
  run: RunStatus | null;
  system: SystemStatus | null;
}) {
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [ci, setCi] = useState<BootstrapReport | null>(null);
  const [live, setLive] = useState<RunProgress | null>(null);
  const [duty, setDuty] = useState<RunStatus["duty"]>(null);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    if (!index) return;
    setResults(null);
    setCi(null);
    api
      .results(index)
      .then(setResults)
      .catch(() => setResults(null));
    api
      .bootstrap(index, 5)
      .then(setCi)
      .catch(() => setCi(null));
  }, [index]);

  // The stream is closed whenever the tab is hidden: a background tab must
  // not keep a measurement machine busy answering questions nobody reads.
  useEffect(() => {
    let source: EventSource | null = null;
    // Opened once regardless of visibility so a background tab still has the
    // current picture; it is closed as soon as the tab is hidden.
    const open = () => {
      if (source) return;
      source = new EventSource("/api/runs/stream");
      source.addEventListener("progress", (event) => {
        const payload = JSON.parse((event as MessageEvent).data);
        setLive(payload);
        setDuty(payload.duty ?? null);
      });
      source.addEventListener("log", (event) => {
        const payload = JSON.parse((event as MessageEvent).data);
        setLogs(payload.out.slice(-40));
      });
    };
    const close = () => {
      source?.close();
      source = null;
    };
    const visibility = () =>
      document.visibilityState === "visible" ? open() : close();
    open();
    document.addEventListener("visibilitychange", visibility);
    return () => {
      document.removeEventListener("visibilitychange", visibility);
      close();
    };
  }, []);

  const active = run?.state === "running" || run?.state === "stopping";
  // A run someone started from a terminal is still a run on this machine.
  const external = run?.state === "external";

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="min-w-0 space-y-4">
        <Plate
          figure={1}
          title="Live progress"
          eyebrow={
            active
              ? "a run is on this machine"
              : external
                ? "a run started outside this server"
                : "nothing running"
          }
          caption={
            active
              ? `stage ${live?.stage ?? run?.argv[4] ?? "?"}, ${count(
                  live?.llm_calls ?? 0,
                )} LLM calls so far`
              : external
                ? `${run?.index} is being written to right now; this server did not start it, so it cannot be stopped from here`
                : "no measurement is running; the stream is idle"
          }
        >
          {active ? (
            <>
              <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="LLM calls" value={count(live?.llm_calls ?? 0)} />
                <Stat
                  label="calls / min"
                  value={(live?.llm_calls_per_min ?? 0).toFixed(1)}
                />
                <Stat label="elapsed" value={seconds(live?.elapsed_s ?? 0)} />
                <Stat
                  label="questions"
                  value={
                    live?.questions_total ? count(live.questions_total) : "—"
                  }
                />
              </dl>
              {duty?.enabled ? (
                <p
                  className="mt-3 text-xs"
                  style={{ color: "var(--color-ink-quiet)" }}
                >
                  duty cycle: cycle {duty.cycle}, {duty.phase} —{" "}
                  {seconds(duty.seconds_left)} left of{" "}
                  {seconds(
                    duty.phase === "window" ? duty.window_s : duty.cool_s,
                  )}
                </p>
              ) : null}
              <pre
                className="mt-3 max-h-40 overflow-auto p-2 text-2xs"
                style={{ background: "var(--color-navy-900)" }}
              >
                {logs.join("\n") || "no output yet"}
              </pre>
              <RunStopBar run={run} />
            </>
          ) : external ? (
            <div>
              <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                <Stat
                  label="LLM calls"
                  value={count(run?.progress?.llm_calls ?? 0)}
                />
                <Stat label="index" value={run?.index ?? "—"} />
                <Stat label="control" value="terminal" />
              </dl>
              <p
                className="mt-3 text-xs"
                style={{ color: "var(--color-ink-quiet)" }}
              >
                Detected from the LLM cache growing on disk, not from a process
                table — so the count is real but the stage and the argv are not
                known here. Starting a second run is blocked while this lasts.
              </p>
            </div>
          ) : (
            <p style={{ color: "var(--color-ink-quiet)" }}>
              Start one below. Progress is read from the LLM cache, which gains
              a line the moment each call returns — that is also why a stopped
              run resumes almost for free.
            </p>
          )}
        </Plate>

        {results ? (
          <Plate
            figure={2}
            title="Results"
            eyebrow={`${count(results.question_count)} questions · S@${results.primary_k} = 0.65·recall + 0.35·novelty`}
            caption={`${index}, written ${results.modified_at.slice(0, 16).replace("T", " ")}`}
          >
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: "var(--color-ink-quiet)" }}>
                  <th className="text-left font-normal">system</th>
                  <th className="text-right font-normal">recall</th>
                  <th className="text-right font-normal">novelty</th>
                  <th className="text-right font-normal">
                    S@{results.primary_k}
                  </th>
                  <th className="text-right font-normal">bridge</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {Object.entries(results.systems).map(([system_, byK]) => {
                  const metrics = byK[String(results.primary_k)];
                  if (!metrics) return null;
                  const isWeb = system_ === "web";
                  return (
                    <tr
                      key={system_}
                      style={{
                        color: isWeb ? "var(--color-energy)" : undefined,
                      }}
                    >
                      <td className="py-1 font-sans">
                        {LABEL[system_] ?? system_}
                      </td>
                      <td className="text-right tabular">
                        {metrics.support_recall.toFixed(4)}
                      </td>
                      <td className="text-right tabular">
                        {metrics.novelty.toFixed(4)}
                      </td>
                      <td className="text-right tabular">
                        {metrics.objective.toFixed(4)}
                      </td>
                      <td className="text-right tabular">
                        {metrics.bridge_recall.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Plate>
        ) : null}

        {ci ? (
          <Plate
            figure={3}
            title="Paired bootstrap"
            eyebrow={`${count(ci.iterations)} resamples, seed ${ci.seed}`}
            caption={`S@${ci.k} differences with 95% intervals; a point estimate alone is not a result`}
          >
            <ul className="space-y-2">
              {ci.diffs.map((diff) => (
                <li key={diff.rival}>
                  <div className="flex items-baseline justify-between text-xs">
                    <span>SPIYWEB − {LABEL[diff.rival] ?? diff.rival}</span>
                    <span
                      className="font-mono tabular"
                      style={{
                        color: diff.significant
                          ? diff.mean > 0
                            ? "var(--color-energy)"
                            : "var(--color-destroyed-ink)"
                          : "var(--color-ink-quiet)",
                      }}
                    >
                      {signed(diff.mean)} CI [{signed(diff.ci_low)},{" "}
                      {signed(diff.ci_high)}] P={diff.p.toFixed(3)}
                      {diff.significant ? "" : " — not significant"}
                    </span>
                  </div>
                  <CiBar
                    low={diff.ci_low}
                    high={diff.ci_high}
                    mean={diff.mean}
                  />
                </li>
              ))}
            </ul>
            <table className="mt-4 w-full text-xs">
              <thead>
                <tr style={{ color: "var(--color-ink-quiet)" }}>
                  <th className="text-left font-normal">hops</th>
                  <th className="text-right font-normal">questions</th>
                  {Object.keys(ci.means).map((system_) => (
                    <th key={system_} className="text-right font-normal">
                      {LABEL[system_] ?? system_}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono">
                {ci.by_hop.map((row) => (
                  <tr key={row.hops}>
                    <td>{row.hops}</td>
                    <td className="text-right tabular">{row.questions}</td>
                    {Object.keys(ci.means).map((system_) => (
                      <td
                        key={system_}
                        className="text-right tabular"
                        style={{
                          color:
                            system_ === "web"
                              ? "var(--color-energy)"
                              : undefined,
                        }}
                      >
                        {(row.scores[system_] ?? 0).toFixed(4)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </Plate>
        ) : null}
      </div>

      <div className="space-y-4">
        <Plate
          figure={4}
          title="System"
          eyebrow="the 88% rule, watched"
          caption={
            system
              ? `VRAM ${share(system.gpu.vram_share)} of ${system.gpu.vram_total_mb} MiB, budget ${share(system.gpu.budget_share)}`
              : "no reading yet"
          }
        >
          {system?.gpu.present ? (
            <>
              <Meter
                ariaLabel="VRAM usage"
                total={system.gpu.vram_total_mb}
                marker={{
                  at: system.gpu.budget_share,
                  label: `budget ${share(system.gpu.budget_share)}`,
                }}
                slices={[
                  {
                    key: "used",
                    label: "in use",
                    value: system.gpu.vram_used_mb,
                    // Blue is occupancy, red is the limit. Painting a healthy
                    // 4% bar red - which is what "energy colour unless over
                    // budget" did after the palette change - made every idle
                    // machine look like it was on fire.
                    color: system.gpu.over_budget
                      ? "var(--color-energy)"
                      : "var(--color-structure)",
                  },
                ]}
              />
              <p
                className="mt-2 text-xs"
                style={{ color: "var(--color-ink-quiet)" }}
              >
                {system.gpu.name} · {system.gpu.utilization}% ·{" "}
                {system.gpu.temperature}
                °C
                {system.gpu.over_budget ? (
                  <strong style={{ color: "var(--color-destroyed-ink)" }}>
                    {" "}
                    — over the 88% budget
                  </strong>
                ) : null}
              </p>
            </>
          ) : (
            <p style={{ color: "var(--color-ink-quiet)" }}>no GPU reading</p>
          )}
          <p
            className="mt-3 text-xs"
            style={{ color: "var(--color-ink-quiet)" }}
          >
            ollama{" "}
            <strong
              style={{
                color: system?.ollama.reachable
                  ? "var(--color-ink)"
                  : "var(--color-destroyed-ink)",
              }}
            >
              {system?.ollama.reachable ? "up" : "down"}
            </strong>
            {system?.ollama.models.length
              ? ` · ${system.ollama.models.slice(0, 3).join(", ")}`
              : ""}
          </p>
        </Plate>

        <RunLauncher
          index={index}
          indexes={indexes}
          disabled={active || external}
        />
      </div>
    </div>
  );
}

function CiBar({
  low,
  high,
  mean,
}: {
  low: number;
  high: number;
  mean: number;
}) {
  const span = Math.max(0.05, Math.abs(low), Math.abs(high)) * 1.2;
  const toPercent = (value: number) => ((value + span) / (2 * span)) * 100;
  return (
    <svg
      viewBox="0 0 100 12"
      className="mt-1 w-full"
      height={12}
      aria-hidden="true"
    >
      <line
        x1="50"
        y1="0"
        x2="50"
        y2="12"
        stroke="var(--color-rule)"
        strokeWidth="1"
        vectorEffect="non-scaling-stroke"
      />
      <line
        x1={toPercent(low)}
        y1="6"
        x2={toPercent(high)}
        y2="6"
        stroke={mean > 0 ? "var(--color-energy)" : "var(--color-destroyed)"}
        strokeWidth="3"
        vectorEffect="non-scaling-stroke"
      />
      <circle
        cx={toPercent(mean)}
        cy="6"
        r="2"
        fill={mean > 0 ? "var(--color-energy)" : "var(--color-destroyed)"}
      />
    </svg>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className="font-display text-xl tabular">{value}</dd>
    </div>
  );
}

function RunStopBar({ run }: { run: RunStatus | null }) {
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  if (!run) return null;
  return (
    <div
      className="mt-3 border-t pt-3"
      style={{ borderColor: "var(--color-rule-hair)" }}
    >
      <p className="eyebrow">running command · pid {run.pid}</p>
      <pre className="overflow-x-auto py-1 text-2xs">{run.argv.join(" ")}</pre>
      <p className="mt-1 text-xs" style={{ color: "var(--color-ink-quiet)" }}>
        Stopping loses only the question in flight: every completed LLM call is
        already on disk, so a restart resumes from the cache.
      </p>
      <div className="mt-2 flex items-center gap-2">
        <input
          value={typed}
          onChange={(event) => setTyped(event.target.value)}
          placeholder="type STOP"
          aria-label="type STOP to confirm"
          className="bg-transparent px-2 py-1 font-mono text-xs"
          style={{ boxShadow: "inset 0 0 0 1px var(--color-destroyed)" }}
        />
        <button
          type="button"
          disabled={typed.trim().toUpperCase() !== "STOP"}
          onClick={() =>
            api
              .runStop(typed)
              .then(() => setTyped(""))
              .catch((failure: unknown) =>
                setError(
                  failure instanceof ApiFailure
                    ? failure.message
                    : String(failure),
                ),
              )
          }
          className="px-3 py-1 font-display text-xs tracking-widest uppercase"
          style={{
            background:
              typed.trim().toUpperCase() === "STOP"
                ? "var(--color-destroyed)"
                : "transparent",
            color:
              typed.trim().toUpperCase() === "STOP"
                ? "var(--color-ink)"
                : "var(--color-ink-quiet)",
            boxShadow: "inset 0 0 0 1px var(--color-destroyed)",
          }}
        >
          stop the run
        </button>
      </div>
      {error ? (
        <p
          className="mt-1 text-xs"
          style={{ color: "var(--color-destroyed-ink)" }}
        >
          {error}
        </p>
      ) : null}
    </div>
  );
}

function RunLauncher({
  index,
  indexes,
  disabled,
}: {
  index: string;
  indexes: IndexSummary[];
  disabled: boolean;
}) {
  const [request, setRequest] = useState<RunRequest>({
    index,
    stage: "report",
    dataset: "musique",
    sample_size: null,
    sample_seed: null,
    force: false,
    no_entity_llm: false,
    propositions: false,
    nli: false,
    skip_iterative: false,
    web: "colored",
    device: "cpu",
    llm_model: null,
    decomp_model: null,
    max_colors: null,
    chain_mode: null,
    duty: { enabled: true, window_s: 2400, cool_s: 300 },
  });
  const [plan, setPlan] = useState<RunPlan | null>(null);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const dataset = indexes.find((row) => row.name === index)?.dataset;
    setRequest((current) => ({
      ...current,
      index,
      dataset: (dataset as RunRequest["dataset"]) ?? current.dataset,
    }));
    setPlan(null);
    setTyped("");
  }, [index, indexes]);

  const ready = useMemo(
    () => plan !== null && typed.trim() === plan.confirm_word,
    [plan, typed],
  );

  return (
    <Plate
      figure={5}
      title="Start a run"
      eyebrow="the plan comes first"
      caption="a run cannot be started from this form; it starts from the plan below"
    >
      <div className="space-y-2 text-xs">
        <label className="flex items-center justify-between gap-2">
          <span className="eyebrow">stage</span>
          <select
            value={request.stage}
            onChange={(event) => {
              setRequest({
                ...request,
                stage: event.target.value as RunRequest["stage"],
              });
              setPlan(null);
            }}
            className="bg-transparent px-2 py-1 font-mono"
            style={{ boxShadow: "inset 0 0 0 1px var(--color-navy-600)" }}
          >
            {["download", "index", "evaluate", "report", "all"].map((stage) => (
              <option
                key={stage}
                value={stage}
                style={{ background: "var(--color-navy-800)" }}
              >
                {stage}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center justify-between gap-2">
          <span className="eyebrow">sample size</span>
          <input
            type="number"
            value={request.sample_size ?? ""}
            // "config default" is clipped in the narrow single-column
            // layout, and a half-read placeholder is worse than a short one.
            placeholder="default"
            onChange={(event) => {
              setRequest({
                ...request,
                sample_size: event.target.value
                  ? Number(event.target.value)
                  : null,
              });
              setPlan(null);
            }}
            className="w-28 bg-transparent px-2 py-1 text-right font-mono"
            style={{ boxShadow: "inset 0 0 0 1px var(--color-navy-600)" }}
          />
        </label>
        {(
          [
            ["propositions", "proposition layer"],
            ["nli", "index-time NLI"],
            ["force", "force rebuild"],
            ["skip_iterative", "skip the iterative baseline"],
          ] as const
        ).map(([key, label]) => (
          <label key={key} className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={Boolean(request[key])}
              onChange={(event) => {
                setRequest({ ...request, [key]: event.target.checked });
                setPlan(null);
              }}
              style={{ accentColor: "var(--color-energy)" }}
            />
            {label}
          </label>
        ))}
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={request.duty.enabled}
            onChange={(event) => {
              setRequest({
                ...request,
                duty: { ...request.duty, enabled: event.target.checked },
              });
              setPlan(null);
            }}
            style={{ accentColor: "var(--color-energy)" }}
          />
          40/5 duty cycle
        </label>
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() =>
          api
            .runPlan(request)
            .then((next) => {
              setPlan(next);
              setError(null);
            })
            .catch((failure: unknown) =>
              setError(
                failure instanceof ApiFailure
                  ? failure.message
                  : String(failure),
              ),
            )
        }
        className="mt-3 w-full px-3 py-2 font-display text-xs tracking-widest uppercase"
        style={{
          boxShadow: "inset 0 0 0 1px var(--color-rule)",
          color: "var(--color-rule)",
          opacity: disabled ? 0.4 : 1,
        }}
      >
        show the plan
      </button>

      {disabled ? (
        <p className="mt-2 text-xs" style={{ color: "var(--color-ink-quiet)" }}>
          A run is already active. One at a time — the server refuses a second
          even if this button is somehow reachable.
        </p>
      ) : null}

      {plan ? (
        <div
          className="mt-3 border-t pt-3 text-xs"
          style={{ borderColor: "var(--color-rule-hair)" }}
        >
          <p className="eyebrow">this exact command will run</p>
          <pre className="overflow-x-auto py-1 text-2xs">
            {plan.argv.join(" ")}
          </pre>
          {plan.will_skip.length ? (
            <p style={{ color: "var(--color-ink-quiet)" }}>
              skips: {plan.will_skip.join(" · ")}
            </p>
          ) : null}
          {plan.estimated_llm_calls ? (
            <p style={{ color: "var(--color-ink-quiet)" }}>
              about {count(plan.estimated_llm_calls)} LLM calls
            </p>
          ) : (
            <p style={{ color: "var(--color-ink-quiet)" }}>
              no call estimate available for this stage
            </p>
          )}
          {plan.warnings.map((warning) => (
            <p key={warning} style={{ color: "var(--color-rule)" }}>
              ! {warning}
            </p>
          ))}
          <div className="mt-2 flex items-center gap-2">
            <input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              placeholder={`type ${plan.confirm_word}`}
              aria-label={`type ${plan.confirm_word} to confirm`}
              className="bg-transparent px-2 py-1 font-mono"
              style={{ boxShadow: "inset 0 0 0 1px var(--color-rule)" }}
            />
            <button
              type="button"
              disabled={!ready}
              onClick={() =>
                api
                  .runStart({ request, token: plan.token, typed })
                  .then(() => {
                    setPlan(null);
                    setTyped("");
                    setError(null);
                  })
                  .catch((failure: unknown) =>
                    setError(
                      failure instanceof ApiFailure
                        ? failure.message
                        : String(failure),
                    ),
                  )
              }
              className="px-3 py-1 font-display tracking-widest uppercase"
              style={{
                background: ready ? "var(--color-energy)" : "transparent",
                color: ready
                  ? "var(--color-navy-900)"
                  : "var(--color-ink-quiet)",
                boxShadow: "inset 0 0 0 1px var(--color-energy)",
              }}
            >
              start
            </button>
          </div>
        </div>
      ) : null}

      {error ? (
        <p
          className="mt-2 text-xs"
          style={{ color: "var(--color-destroyed-ink)" }}
        >
          {error}
        </p>
      ) : null}
    </Plate>
  );
}
