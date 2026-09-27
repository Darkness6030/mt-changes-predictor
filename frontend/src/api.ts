/** One thin API layer. The UI never computes risk or picks a target: it renders Backend data. */

import type {
  AuthState, DemoSource, DemoState, ExplanationAnswer, WhatIfAnswer, HistoryFrame, Quality, Snapshot, Status, VehicleDetail,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

const TOKEN_KEY = "dispatcher-token";

/** Optional sign-in token (the stand runs without accounts unless the Backend enables them). */
export const authToken = {
  get(): string | null {
    try { return window.localStorage.getItem(TOKEN_KEY); } catch { return null; }
  },
  set(token: string): void {
    try { window.localStorage.setItem(TOKEN_KEY, token); } catch { /* private mode: session only */ }
    memoryToken = token;
  },
  clear(): void {
    try { window.localStorage.removeItem(TOKEN_KEY); } catch { /* nothing stored */ }
    memoryToken = null;
  },
};
let memoryToken: string | null = null;

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = memoryToken ?? authToken.get();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

/** A 401 means the token expired or was revoked: the sign-in gate asks again. */
function checkAuth(response: Response): void {
  if (response.status === 401) window.dispatchEvent(new Event("auth-required"));
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, 5000);
  try {
    const response = await fetch(path, { signal: controller.signal, headers: authHeaders({ Accept: "application/json" }) });
    checkAuth(response);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new ApiError(body?.detail?.detail ?? `${response.status} ${response.statusText}`, response.status);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (timedOut) throw new ApiError("Ответ сервера не получен за 5 секунд");
    throw error;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

export type ReplayAction = "start" | "pause" | "reset" | "speed" | "seek";

export const api = {
  auth: (signal?: AbortSignal) => request<AuthState>("/api/v1/auth", signal),
  history: (runId: string, at: string, signal?: AbortSignal) =>
    request<HistoryFrame>(`/api/v1/history?${new URLSearchParams({ run_id: runId, at })}`, signal),
  demo: (signal?: AbortSignal) => request<DemoState>("/api/v1/demo/sources", signal),
  async source(source: DemoSource, speed: number): Promise<DemoState> {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch("/api/v1/demo/source", {
        method: "POST", signal: controller.signal,
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ source, speed }),
      });
      checkAuth(response);
      const body = await response.json();
      if (!response.ok) throw new ApiError(body?.detail?.detail ?? "Источник не переключён", response.status);
      return body as DemoState;
    } catch (error) {
      if (controller.signal.aborted) throw new ApiError("Переключение ещё не подтверждено. Проверяется состояние Backend.");
      throw error;
    } finally { window.clearTimeout(timer); }
  },
  snapshot: (signal?: AbortSignal) => request<Snapshot>("/api/v1/snapshot", signal),
  status: (signal?: AbortSignal) => request<Status>("/api/v1/status", signal),
  quality: (signal?: AbortSignal) => request<Quality>("/api/v1/metrics/quality", signal),
  vehicle: (trId: string, signal?: AbortSignal) =>
    request<VehicleDetail>(`/api/v1/vehicles/${encodeURIComponent(trId)}`, signal),
  whatif: (trId: string, reserveInMin: number, signal?: AbortSignal) =>
    request<WhatIfAnswer>(
      `/api/v1/vehicles/${encodeURIComponent(trId)}/whatif?${new URLSearchParams({ reserve_in_min: String(reserveInMin) })}`,
      signal,
    ),
  explanation: (trId: string, predictionId: string, signal?: AbortSignal) =>
    request<ExplanationAnswer>(
      `/api/v1/vehicles/${encodeURIComponent(trId)}/explanation?${new URLSearchParams({ prediction_id: predictionId })}`,
      signal,
    ),
  async acknowledge(alertId: string): Promise<void> {
    const response = await fetch(`/api/v1/alerts/${encodeURIComponent(alertId)}/ack`, {
      method: "POST", headers: authHeaders(),
    });
    checkAuth(response);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new ApiError(body?.detail?.detail ?? "Не удалось отметить алерт", response.status);
    }
  },
  async replay(action: ReplayAction, speed?: number, startAt?: string): Promise<void> {
    const response = await fetch("/api/v1/replay/control", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ action, speed, start_at: startAt }),
    });
    checkAuth(response);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new ApiError(body?.detail?.detail ?? "Команда replay отклонена", response.status);
    }
  },
};
