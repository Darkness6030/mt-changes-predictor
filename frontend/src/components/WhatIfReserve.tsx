import { useEffect, useState } from "react";
import { api } from "../api";
import { signedDelay, sourceTime } from "../format";
import type { WhatIfAnswer } from "../types";

const OPTIONS = [5, 10, 15, 20, 30];

interface Props {
  trId: string;
  predictionId: string;
}

/** What-if: a reserve vehicle reaches the terminal and takes over the line's next trips. */
export function WhatIfReserve({ trId, predictionId }: Props) {
  const [minutes, setMinutes] = useState(10);
  const [answer, setAnswer] = useState<WhatIfAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    api.whatif(trId, minutes, controller.signal)
      .then(setAnswer)
      .catch((failure: Error) => { if (!controller.signal.aborted) setError(failure.message); });
    return () => controller.abort();
  }, [trId, predictionId, minutes]);

  return (
    <div className="section">
      <h3>What-if: выпустить резервное ТС</h3>
      <label className="whatif-control" htmlFor="whatif-minutes">
        Резерв на конечной через
        <select
          id="whatif-minutes"
          value={minutes}
          onChange={(event) => setMinutes(Number(event.target.value))}
        >
          {OPTIONS.map((value) => <option key={value} value={value}>{value} мин</option>)}
        </select>
      </label>
      {error ? <div className="hint">{error}</div> : null}
      {answer && !answer.available ? (
        <div className="hint">
          {answer.reason === "no_following_trips"
            ? "После текущего рейса у линии нет плановых рейсов: резерв не нужен."
            : "Расчёт доступен для ТС с текущим прогнозом и определённым рейсом."}
        </div>
      ) : null}
      {answer?.available && answer.trips ? (
        <>
          <div className={answer.late_trips_with! < answer.late_trips_without! ? "advice" : "hint"}>
            {answer.late_trips_without
              ? `Без резерва: ${answer.late_trips_without} рейс(а) с опозданием > 2 мин; с резервом: ${answer.late_trips_with}.`
              : "Опоздание гасится отстоем на конечной: следующие рейсы уходят вовремя и без резерва."}
            {answer.reserve_takes_trip ? ` Резерв берёт рейс ${answer.reserve_takes_trip} из ${answer.trips_total}.` : ""}
          </div>
          <table className="prediction-history whatif-table">
            <thead>
              <tr><th>Рейс</th><th>План</th><th>Без резерва</th><th>С резервом</th></tr>
            </thead>
            <tbody>
              {answer.trips.map((row) => (
                <tr key={row.trip}>
                  <td>{row.trip}</td>
                  <td>{sourceTime(row.planned_start_at)}</td>
                  <td className={row.delay_without_s > 120 ? "risk-red" : undefined}>
                    {signedDelay(row.delay_without_s)}
                  </td>
                  <td className={row.delay_with_s > 120 ? "risk-red" : "risk-green"}>
                    {signedDelay(row.delay_with_s)}{row.served_by === "reserve" ? " · резерв" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="hint">Допущения: {answer.assumptions?.join("; ")}.</div>
        </>
      ) : null}
    </div>
  );
}
