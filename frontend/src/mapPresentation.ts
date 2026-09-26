import { duration, signedDelay } from "./format";
import type { CurrentDeviation, Vehicle } from "./types";

export const RISK_COLOURS: Record<string, string> = {
  green: "#2fbf71", yellow: "#f2b63c", red: "#ff5d5d", none: "#6b7a8d",
};

export function forecastColour(vehicle: Vehicle): string {
  const prediction = vehicle.prediction;
  return prediction && ["ok", "ml_unavailable"].includes(prediction.status) && prediction.risk_level
    ? RISK_COLOURS[prediction.risk_level] : RISK_COLOURS.none;
}

export function deviationColour(deviation?: CurrentDeviation): string {
  return deviation?.status === "ok" && deviation.risk_level
    ? RISK_COLOURS[deviation.risk_level] : RISK_COLOURS.none;
}

export function deviationNote(deviation?: CurrentDeviation): string {
  if (!deviation || deviation.status === "unavailable") return "Нет сопоставления GPS с остановкой";
  switch (deviation.reason) {
    case "stale_gps": return "Позиция ТС устарела";
    case "no_valid_position": return "Нет достоверной позиции ТС";
    case "estimate_too_old": return `Оценка старше ${duration(deviation.max_age_s)}`;
    default: return "Оценка по GPS на последней распознанной остановке";
  }
}

export function deviationTooltip(deviation?: CurrentDeviation): string {
  if (!deviation || deviation.status === "unavailable") return "Текущее отклонение: нет оценки";
  return `Текущее (оценка GPS): ${signedDelay(deviation.delay_s)}` +
    ` · ${duration(deviation.age_s)} назад` +
    (deviation.status === "stale" ? ` · ${deviationNote(deviation)}` : "");
}
