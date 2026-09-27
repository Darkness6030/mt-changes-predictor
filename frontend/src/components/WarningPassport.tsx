import { duration, signedDelay, sourceTime, stopLabel } from "../format";
import type { Alert, VehicleDetail } from "../types";

interface Props {
  detail: VehicleDetail;
  alerts: Alert[];
}

/**
 * When the dispatcher was warned, how the forecast evolved and what actually happened.
 * The outcome comes from received GPS after the planned time, never from labels.
 */
export function WarningPassport({ detail, alerts }: Props) {
  const prediction = detail.prediction;
  const own = alerts.filter((alert) => alert.tr_id === detail.tr_id);
  const current = own.find(
    (alert) => alert.state === "active" && alert.target_stop_id === prediction?.target_stop_id,
  );
  const outcomes = own
    .filter((alert) => alert.observed_arrival_at)
    .sort((a, b) => (a.target_planned_at < b.target_planned_at ? 1 : -1))
    .slice(0, 3);
  if (!current && !outcomes.length) return null;
  const history = current
    ? detail.prediction_history
        .filter((item) => item.target_stop_id === current.target_stop_id)
        .slice(-6)
    : [];
  return (
    <div className="section">
      <h3>Паспорт предупреждения</h3>
      {current ? (
        <ol className="passport">
          <li>
            <span>{sourceTime(current.first_alert_at)}</span>
            <b>Первое предупреждение</b>
            <em>за {duration(current.planned_lead_s)} до планового прибытия</em>
          </li>
          {history.length > 1 ? (
            <li>
              <span>прогноз</span>
              <b className="passport-trend">
                {history.map((item) => signedDelay(item.delay_s)).join(" → ")}
              </b>
              <em>последние пересчёты к этой цели</em>
            </li>
          ) : null}
          <li>
            <span>{sourceTime(current.target_planned_at)}</span>
            <b>Плановое прибытие</b>
            <em>
              ожидается {signedDelay(current.delay_s)} · {stopLabel(null, current.target_stop_id)}
            </em>
          </li>
          <li className="passport-pending">
            <span>факт</span>
            <b>Ещё не наступил</b>
            <em>появится по GPS после прибытия ТС к остановке</em>
          </li>
        </ol>
      ) : null}
      {outcomes.length ? (
        <>
          <div className="hint passport-caption">Что произошло после прошлых предупреждений</div>
          <ul className="passport-outcomes">
            {outcomes.map((alert) => {
              const confirmed = Math.abs(alert.observed_delay_s ?? 0) > 60;
              return (
                <li key={alert.alert_id}>
                  <span className={confirmed ? "badge stale" : "badge"}>
                    {confirmed ? "подтвердилось" : "обошлось"}
                  </span>
                  <span>
                    предупредили {sourceTime(alert.first_alert_at)} → прибыл{" "}
                    {sourceTime(alert.observed_arrival_at)} ({signedDelay(alert.observed_delay_s)})
                    · за {duration(alert.warning_lead_s)} до прибытия
                  </span>
                </li>
              );
            })}
          </ul>
        </>
      ) : null}
    </div>
  );
}
