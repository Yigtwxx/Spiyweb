import type {
  AtomHit,
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
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
