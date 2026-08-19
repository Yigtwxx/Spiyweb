import { useEffect, useState } from "react";

import { api } from "./lib/api";
import { share } from "./lib/format";
import type {
  IndexDetail,
  IndexSummary,
  RunStatus,
  SystemStatus,
} from "./lib/types";
import { InspectView } from "./views/InspectView";
import { RunsView } from "./views/RunsView";

type View = "inspect" | "runs";

export default function App() {
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
