/** Formatting helpers. Source timestamps keep their basis: no timezone is applied. */

import type { Prediction, RiskLevel } from "./types";

export const RISK_LABEL: Record<RiskLevel, string> = {
  green: "В графике",
  yellow: "Внимание",
  red: "Опоздание",
};

export const RISK_SIGN: Record<RiskLevel, string> = { green: "●", yellow: "▲", red: "■" };

export const STATUS_LABEL: Record<string, string> = {
  ok: "Прогноз есть",
  warming_up: "Недостаточно истории",
  no_target_in_horizon: "Нет плановой остановки в горизонте",
  no_schedule: "Нет расписания для ТС",
  no_mapping: "Устройство не сопоставлено",
  stale: "Данные устарели",
  ml_unavailable: "ML недоступен",
  invalid_input: "Признаки не прошли проверку",
};

export const HINT_LABEL: Record<string, string> = {
  supplied: "подсказка из прогнозной точки",
  estimated: "оценка по GPS и плану",
  missing: "недоступна, модель без подсказки",
};

/** Signed minutes and seconds; the sign is always explicit for a dispatcher. */
export function signedDelay(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const sign = seconds >= 0 ? "+" : "−";
  const total = Math.abs(Math.round(seconds));
  if (total < 60) return `${sign}${total} с`;
  return `${sign}${Math.floor(total / 60)} мин ${String(total % 60).padStart(2, "0")} с`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total} с`;
  if (total < 3600) return `${Math.floor(total / 60)} мин ${String(total % 60).padStart(2, "0")} с`;
  return `${Math.floor(total / 3600)} ч ${String(Math.floor((total % 3600) / 60)).padStart(2, "0")} мин`;
}

/** Time part of a dataset-naive timestamp, without inventing an offset. */
export function sourceTime(value: string | null | undefined): string {
  if (!value) return "—";
  const match = value.match(/\d{2}:\d{2}:\d{2}/);
  return match ? match[0] : value;
}

export function sourceDate(value: string | null | undefined): string {
  if (!value) return "—";
  return value.slice(0, 10);
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits).replace(".", ",")} %`;
}

export function number(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits).replace(".", ",");
}

export function freshness(vehicle: { stale: boolean; position: unknown }): {
  label: string;
  tone: string;
} {
  if (!vehicle.position) return { label: "нет позиции", tone: "missing" };
  if (vehicle.stale) return { label: "устаревшие данные", tone: "stale" };
  return { label: "данные свежие", tone: "fresh" };
}

/** Severity for the queue: red first, then yellow, then everything without a forecast. */
/** Queue order: red, then a first/last trip at risk, yellow, probability-only, the rest. */
export function severity(prediction: Prediction | null): number {
  if (!prediction || prediction.status !== "ok") return 1;
  const edge = Boolean(prediction.trip?.first || prediction.trip?.last);
  if (prediction.risk_level === "red") return edge ? 7 : 6;
  if (prediction.risk_level === "yellow") return edge ? 5 : 4;
  if (prediction.attention === "probability") return edge ? 5 : 3;
  return 2;
}

/** "первый рейс" / "последний рейс" label for the target visit, if it is a day's edge. */
export function tripEdge(prediction: Prediction | null | undefined): string | null {
  if (prediction?.trip?.first) return "первый рейс";
  if (prediction?.trip?.last) return "последний рейс";
  return null;
}

/** A planned-visit label a dispatcher can read when the dataset has no address. */
export function stopLabel(address: string | null | undefined, visitId: string | null | undefined): string {
  if (address) return address;
  return visitId ? `остановка без адреса · №…${visitId.slice(-4)}` : "остановка без адреса";
}
