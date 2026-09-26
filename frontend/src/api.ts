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
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, 5000);
  try {
    const response = await fetch(path, { signal: controller.signal, headers: { Accept: "application/json" } });
    if (!response.ok) throw new ApiError(`${response.status} ${response.statusText}`, response.status);
    return (await response.json()) as T;
  } catch (error) {
    if (timedOut) throw new ApiError("Ответ сервера не получен за 5 секунд");
    throw error;
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
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
