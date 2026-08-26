import type {
  AtomHit,
  Capabilities,
  BootstrapReport,
  IndexDetail,
  IndexSummary,
  InspectRequest,
  InspectResponse,
  LogTail,
  ResultsPayload,
  RunPlan,
  RunRequest,
  RunStatus,
  SystemStatus,
  TraceListDto,
  TraceRecordDto,
  SceneDto,
} from "./types";

/** A failed request, carrying the server's own explanation. */
export class ApiFailure extends Error {
  // Written out rather than declared as constructor parameter properties:
  // `erasableSyntaxOnly` forbids the shorthand, and this file must stay
  // strippable by the bundler without a TypeScript transform.
  status: number;
  hint: string | null;

  constructor(status: number, message: string, hint: string | null = null) {
    super(message);
    this.name = "ApiFailure";
    this.status = status;
    this.hint = hint;
  }
}

/**
 * The viewer's process token, taken from the URL once and kept in memory.
 *
 * `spiyweb.viewer` hands out `http://127.0.0.1:PORT/?token=...` and guards
 * every `/api/*` route with it. The token moves out of the query string and
 * into a header on the first request: a URL is copied into chat windows and
 * written to proxy logs, a header is not, and the address bar keeps showing
 * a link that still works if someone does copy it.
 *
 * The repository rig serves no token and needs none - it is bound to a
 * developer's own machine and predates this - so an absent token is not an
 * error here. The server is the one that decides.
 */
const TOKEN: string | null = (() => {
  const params = new URLSearchParams(window.location.search);
  const found = params.get("token");
  if (!found) return null;
  params.delete("token");
  const rest = params.toString();
  window.history.replaceState(
    null,
    "",
    `${window.location.pathname}${rest ? `?${rest}` : ""}`,
  );
  return found;
})();

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(TOKEN ? { "X-Spiyweb-Token": TOKEN } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    // FastAPI puts a plain string in `detail`; our own errors send a shape.
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
      message?: string;
      hint?: string | null;
    } | null;
    throw new ApiFailure(
      response.status,
      body?.message ?? body?.detail ?? `request failed (${response.status})`,
      body?.hint ?? null,
    );
  }
  return (await response.json()) as T;
}

export const api = {
  capabilities: () => request<Capabilities>("/api/capabilities"),
  traces: (limit = 50, offset = 0) =>
    request<TraceListDto>(`/api/traces?limit=${limit}&offset=${offset}`),
  trace: (id: string) => request<TraceRecordDto>(`/api/traces/${id}`),
  traceScene: (id: string, maxNodes = 300) =>
    request<SceneDto>(`/api/traces/${id}/scene?max_nodes=${maxNodes}`),
  ask: (body: { query?: string; parts?: Record<string, string>; profile?: string | null }) =>
    request<TraceRecordDto>("/api/query", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  indexes: () => request<IndexSummary[]>("/api/indexes"),
  index: (name: string) => request<IndexDetail>(`/api/indexes/${name}`),
  atoms: (index: string, q: string, signal?: AbortSignal) =>
    request<AtomHit[]>(
      `/api/atoms?index=${encodeURIComponent(index)}&q=${encodeURIComponent(q)}`,
      { signal },
    ),
  inspect: (body: InspectRequest, signal?: AbortSignal) =>
    request<InspectResponse>("/api/inspect", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  system: () => request<SystemStatus>("/api/system"),
  clearCache: () => request<unknown>("/api/cache", { method: "DELETE" }),
  results: (name: string) => request<ResultsPayload>(`/api/results/${name}`),
  bootstrap: (name: string, k = 5) =>
    request<BootstrapReport>(`/api/results/${name}/ci?k=${k}`),
  runCurrent: () => request<RunStatus>("/api/runs/current"),
  runPlan: (body: RunRequest) =>
    request<RunPlan>("/api/runs/plan", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  runStart: (body: { request: RunRequest; token: string; typed: string }) =>
    request<RunStatus>("/api/runs/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  runStop: (typed: string) =>
    request<RunStatus>("/api/runs/stop", {
      method: "POST",
      body: JSON.stringify({ typed }),
    }),
  runRelease: () => request<RunStatus>("/api/runs/release", { method: "POST" }),
  logs: (tail = 200) => request<LogTail>(`/api/runs/logs?tail=${tail}`),
};
