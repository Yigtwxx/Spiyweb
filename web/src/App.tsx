import { useEffect, useState } from "react";

import { api } from "./lib/api";
import { share } from "./lib/format";
import type {
  Capabilities,
  IndexDetail,
  IndexSummary,
  RunStatus,
  SystemStatus,
} from "./lib/types";
import { InspectView } from "./views/InspectView";
import { RunsView } from "./views/RunsView";
import { TracesView } from "./views/TracesView";

type View = "inspect" | "runs";

/**
 * One bundle, two products.
 *
 * The same compiled page is served by two very different processes: the
 * repository's measurement rig, which owns `data/` and supervises benchmark
 * runs, and `spiyweb.viewer`, which ships inside the wheel and shows an
 * application its own recorded calls. Faz 2.5 made the second one exist, and
 * building a second front end for it would have meant two copies of the
 * canvas, the ledger strip and the API client drifting apart.
 *
 * So the page ASKS. `/api/capabilities` is answered by both servers, and the
 * shell below is chosen from the answer — never by probing an endpoint and
 * inferring from a 404, which stops being true the day a route is renamed.
 */
export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    api
      .capabilities()
      .then(setCapabilities)
      .catch((error: Error) => setFailed(error.message));
  }, []);

  if (failed)
    return (
      <main className="mx-auto max-w-[48rem] px-4 py-16">
        <h1 className="font-display text-2xl tracking-[0.22em]">SPIYWEB</h1>
        <p className="font-plate mt-4" role="alert">
          This page could not reach its server: {failed}
        </p>
        <p
          className="font-plate mt-2 italic"
          style={{ color: "var(--color-ink-quiet)" }}
        >
          If you opened a bare address, use the link the viewer printed — it
          carries the token that every API route here requires.
        </p>
      </main>
    );
  if (!capabilities) return null;
  return capabilities.runs ? (
    <RigShell />
  ) : (
    <ViewerShell capabilities={capabilities} />
  );
}

/**
 * The wheel's viewer: recorded calls, and a question box where there is an
 * index behind the page. No index picker and no run controls, because this
 * process has exactly one index and supervises nothing.
 */
function ViewerShell({ capabilities }: { capabilities: Capabilities }) {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header
        className="sticky top-0 z-10 border-b"
        style={{
          borderColor: "var(--color-rule-hair)",
          background:
            "color-mix(in oklab, var(--color-navy-900) 88%, transparent)",
          backdropFilter: "blur(6px)",
        }}
      >
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-end gap-x-6 gap-y-2 px-4 py-3">
          <div>
            <h1
              className="font-display text-2xl leading-none tracking-[0.22em]"
              style={{ color: "var(--color-ink)" }}
            >
              SPIYWEB
            </h1>
            <p
              className="font-plate italic"
              style={{ color: "var(--color-rule)", fontSize: "var(--text-sm)" }}
            >
              {capabilities.live
                ? "what this application retrieved — and what it would retrieve"
                : "what this application retrieved"}
            </p>
          </div>
          <div
            className="ml-auto flex items-center gap-4 font-mono text-xs"
            style={{ color: "var(--color-ink-quiet)" }}
          >
            <span title={capabilities.origin}>
              {capabilities.mode === "file" ? "trace file" : "live index"}
            </span>
            <span>v{capabilities.version}</span>
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-[1600px] px-4 py-4">
        <TracesView capabilities={capabilities} />
      </main>
    </>
  );
}

/** The repository rig: index picker, live inspection, measurement runs. */
function RigShell() {
  const [indexes, setIndexes] = useState<IndexSummary[]>([]);
  const [index, setIndex] = useState<string>("");
  const [detail, setDetail] = useState<IndexDetail | null>(null);
  const [view, setView] = useState<View>("inspect");
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [run, setRun] = useState<RunStatus | null>(null);

  useEffect(() => {
    api.indexes().then((found) => {
      setIndexes(found);
      const params = new URLSearchParams(window.location.search);
      const wanted = params.get("index");
      setIndex(
        wanted && found.some((row) => row.name === wanted)
          ? wanted
          : (found[0]?.name ?? ""),
      );
      const wantedView = params.get("view");
      if (wantedView === "runs" || wantedView === "inspect")
        setView(wantedView);
    });
  }, []);

  useEffect(() => {
    if (!index) return;
    api
      .index(index)
      .then(setDetail)
      .catch(() => setDetail(null));
    const params = new URLSearchParams(window.location.search);
    params.set("index", index);
    params.set("view", view);
    window.history.replaceState(null, "", `?${params.toString()}`);
  }, [index, view]);

  // The system pill is the only thing this page polls; everything that
  // streams goes through SSE. The FIRST read is unconditional - a tab that
  // starts in the background would otherwise show "ollama down" until it is
  // focused, which is a lie about the machine. Repeat polling is what the
  // visibility check guards, because that is the part that would keep asking
  // a measurement machine questions nobody is reading.
  useEffect(() => {
    const read = () => {
      api
        .system()
        .then(setSystem)
        .catch(() => undefined);
      api
        .runCurrent()
        .then(setRun)
        .catch(() => undefined);
    };
    const poll = () => {
      if (document.visibilityState === "visible") read();
    };
    read();
    const timer = window.setInterval(poll, 10000);
    document.addEventListener("visibilitychange", poll);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", poll);
    };
  }, []);

  const runActive = run?.state === "running" || run?.state === "stopping";
  // An outside run still owns the machine, so inspection still stands aside.
  const machineBusy = runActive || run?.state === "external";

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header
        className="sticky top-0 z-10 border-b"
        style={{
          borderColor: "var(--color-rule-hair)",
          background:
            "color-mix(in oklab, var(--color-navy-900) 88%, transparent)",
          backdropFilter: "blur(6px)",
        }}
      >
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-end gap-x-6 gap-y-2 px-4 py-3">
          <div>
            <h1
              className="font-display text-2xl leading-none tracking-[0.22em]"
              style={{ color: "var(--color-ink)" }}
            >
              SPIYWEB
            </h1>
            <p
              className="font-plate italic"
              style={{ color: "var(--color-rule)", fontSize: "var(--text-sm)" }}
            >
              spreading-activation rig — phase 1
            </p>
          </div>

          <label className="flex items-center gap-2">
            <span className="eyebrow">index</span>
            <select
              value={index}
              onChange={(event) => setIndex(event.target.value)}
              className="bg-transparent px-2 py-1 font-mono text-xs"
              style={{ boxShadow: "inset 0 0 0 1px var(--color-navy-600)" }}
            >
              {indexes.map((row) => (
                <option
                  key={row.name}
                  value={row.name}
                  style={{ background: "var(--color-navy-800)" }}
                >
                  {row.name} · {row.corpus_chunks ?? "?"} chunks
                </option>
              ))}
            </select>
          </label>

          <div role="tablist" aria-label="views" className="flex gap-1">
            {(["inspect", "runs"] as const).map((name) => (
              <button
                key={name}
                role="tab"
                aria-selected={view === name}
                onClick={() => setView(name)}
                className="px-3 py-1 font-display text-xs tracking-widest uppercase"
                style={{
                  color:
                    view === name
                      ? "var(--color-navy-900)"
                      : "var(--color-ink)",
                  background:
                    view === name ? "var(--color-rule)" : "transparent",
                  boxShadow:
                    view === name
                      ? "none"
                      : "inset 0 0 0 1px var(--color-rule-hair)",
                }}
              >
                {name}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-4 font-mono text-xs">
            {system?.gpu.present ? (
              <span
                title={`${system.gpu.name} · ${system.gpu.temperature ?? "?"}°C`}
                style={{
                  color: system.gpu.over_budget
                    ? "var(--color-destroyed-ink)"
                    : "var(--color-ink-quiet)",
                }}
              >
                VRAM {share(system.gpu.vram_share)} · GPU{" "}
                {system.gpu.utilization}%
              </span>
            ) : null}
            <span
              style={{
                color: system?.ollama.reachable
                  ? "var(--color-ink-quiet)"
                  : "var(--color-destroyed-ink)",
              }}
            >
              ollama {system?.ollama.reachable ? "up" : "down"}
            </span>
            {machineBusy ? (
              <span style={{ color: "var(--color-energy)" }}>
                ● {runActive ? "run active" : "outside run"} — {run?.index}
              </span>
            ) : null}
          </div>
        </div>
      </header>

      <main id="main" className="mx-auto max-w-[1600px] px-4 py-4">
        {view === "inspect" ? (
          <InspectView
            index={index}
            detail={detail}
            runActive={Boolean(machineBusy)}
          />
        ) : (
          <RunsView index={index} indexes={indexes} run={run} system={system} />
        )}
      </main>
    </>
  );
}
