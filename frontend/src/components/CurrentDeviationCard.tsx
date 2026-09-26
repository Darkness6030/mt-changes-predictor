import { duration, signedDelay, sourceTime } from "../format";
import { deviationNote } from "../mapPresentation";
import type { CurrentDeviation } from "../types";

export function CurrentDeviationCard({ value }: { value?: CurrentDeviation }) {
  const available = value?.delay_s != null;
  const tone = value?.status === "ok" ? value.risk_level ?? "none" : "none";
  return (
    <div className="section current-deviation" aria-label="Текущее отклонение по GPS">
      <h3>Текущее отклонение · обводка ТС</h3>
      <div className={`current-deviation-value risk-${tone}`}>
        {available ? signedDelay(value!.delay_s) : "Нет оценки"}
        {value?.status === "stale" ? <span className="badge stale">устарело</span> : null}
      </div>
      <div className="hint">{deviationNote(value)}</div>
      {available ? <>
        <div className="hint">Наблюдение {sourceTime(value!.observed_at)} · {duration(value!.age_s)} назад</div>
        <div className="hint">{value!.visit_address ?? `Посещение ${value!.visit_id}`}</div>
      </> : null}
    </div>
  );
}
