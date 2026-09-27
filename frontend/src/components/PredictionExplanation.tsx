import { useEffect, useState } from "react";
import { api } from "../api";
import { signedDelay } from "../format";
import type { Explanation } from "../types";

type State =
  | { kind: "loading" }
  | { kind: "ready"; value: Explanation }
  | { kind: "error"; message: string };

interface Props {
  trId: string;
  predictionId: string;
  /** History frames are read-only snapshots: their feature rows are no longer kept. */
  readOnly?: boolean;
}

/** Additive split of the opened forecast, requested once per prediction id. */
export function PredictionExplanation({ trId, predictionId, readOnly = false }: Props) {
  const [state, setState] = useState<State>({ kind: "loading" });
  useEffect(() => {
    if (readOnly) return;
    const controller = new AbortController();
    setState({ kind: "loading" });
    api.explanation(trId, predictionId, controller.signal)
      .then((answer) => setState({ kind: "ready", value: answer.explanation }))
      .catch((error: Error) => {
        if (!controller.signal.aborted) setState({ kind: "error", message: error.message });
      });
    return () => controller.abort();
  }, [trId, predictionId, readOnly]);

  return (
    <div className="section">
      <h3>Из чего сложился прогноз</h3>
      {readOnly ? (
        <div className="hint">Разложение доступно для текущего прогноза, не для истории.</div>
      ) : state.kind === "loading" ? (
        <div className="hint">Считаем вклады признаков…</div>
      ) : state.kind === "error" ? (
        <div className="hint">{state.message}</div>
      ) : (
        <Bars value={state.value} />
      )}
    </div>
  );
}

function Bars({ value }: { value: Explanation }) {
  const parts = [
    ...value.groups.map((group) => ({ key: group.group, label: group.label, seconds: group.seconds })),
    ...(Math.abs(value.other_s) >= 0.5
      ? [{ key: "other", label: "Остальные признаки", seconds: value.other_s }]
      : []),
  ];
  const scale = Math.max(1, ...parts.map((part) => Math.abs(part.seconds)));
  return (
    <>
      <ul className="explain" aria-label="Вклады в прогноз, секунды">
        <li className="explain-base">
          <span>Типичная поправка модели</span>
          <b>{signedDelay(value.base_s)}</b>
        </li>
        {parts.map((part) => (
          <li key={part.key}>
            <span>{part.label}</span>
            <i className="explain-track" aria-hidden="true">
              <i
                className={part.seconds >= 0 ? "explain-bar late" : "explain-bar early"}
                style={{ width: `${(Math.abs(part.seconds) / scale) * 100}%` }}
              />
            </i>
            <b>{signedDelay(part.seconds)}</b>
          </li>
        ))}
        <li className="explain-total">
          <span>Итого прогноз</span>
          <b>{signedDelay(value.total_s)}</b>
        </li>
      </ul>
      <div className="hint">
        Вклады групп признаков в секундах (SHAP, точная сумма). Это арифметика модели, а не
        установленная причина: красный тянет к опозданию, зелёный — к опережению.
      </div>
    </>
  );
}
