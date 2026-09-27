import { signedDelay } from "../format";
import type { Explanation } from "../types";

/** Additive split of one forecast; bar length is relative to the largest part. */
export function PredictionExplanation({ value }: { value?: Explanation | null }) {
  if (!value) return null;
  const parts = [
    ...value.groups.map((group) => ({ key: group.group, label: group.label, seconds: group.seconds })),
    ...(Math.abs(value.other_s) >= 0.5
      ? [{ key: "other", label: "Остальные признаки", seconds: value.other_s }]
      : []),
  ];
  const scale = Math.max(1, ...parts.map((part) => Math.abs(part.seconds)));
  return (
    <div className="section">
      <h3>Из чего сложился прогноз</h3>
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
    </div>
  );
}
