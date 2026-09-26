/** One thin API layer. The UI never computes risk or picks a target: it renders Backend data. */

import type { Quality, Snapshot, Status, VehicleDetail } from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body?.detail?.detail ?? body?.detail ?? detail;
    } catch {
      /* keep the status line */
    }
    throw new ApiError(String(detail), response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  snapshot: (signal?: AbortSignal) => request<Snapshot>("/api/v1/snapshot", signal),
  status: (signal?: AbortSignal) => request<Status>("/api/v1/status", signal),
  quality: (signal?: AbortSignal) => request<Quality>("/api/v1/metrics/quality", signal),
  vehicle: (trId: string, signal?: AbortSignal) =>
    request<VehicleDetail>(`/api/v1/vehicles/${encodeURIComponent(trId)}`, signal),
  async acknowledge(alertId: string): Promise<void> {
    const response = await fetch(`/api/v1/alerts/${encodeURIComponent(alertId)}/ack`, {
      method: "POST",
    });
    if (!response.ok) throw new ApiError("Не удалось отметить алерт", response.status);
  },
  async replay(action: "start" | "pause" | "reset" | "speed", speed?: number): Promise<void> {
    const response = await fetch("/api/v1/replay/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(speed === undefined ? { action } : { action, speed }),
    });
    if (!response.ok) throw new ApiError("Команда replay отклонена", response.status);
  },
};
