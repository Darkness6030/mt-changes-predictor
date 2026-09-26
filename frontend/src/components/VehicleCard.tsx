import {
  HINT_LABEL,
  RISK_LABEL,
  STATUS_LABEL,
  duration,
  freshness,
  number,
  percent,
  signedDelay,
  sourceTime,
} from "../format";
import { CurrentDeviationCard } from "./CurrentDeviationCard";
import type { Alert, RiskPolicy, VehicleDetail } from "../types";

interface Props {
  readOnly?: boolean;
  detail: VehicleDetail | null;
  alerts: Alert[];
  policy: RiskPolicy | null;
  onAcknowledge: (alertId: string) => void;
}

const CALIBRATION_NOTE: Record<string, string> = {
  validated: "вероятность проверена на отложенной выборке",
  fitted_on_development: "калибровка подобрана на development-фолде того же дня",
  weak: "калибровка слабая, вероятность использовать осторожно",
  unavailable: "модель вероятности отсутствует",
};

/** Timeline T → +10 мин (исключено) → цель → +15 мин (включено). */
function Timeline({ horizon }: { horizon: number }) {
  const position = Math.min(100, Math.max(0, ((horizon - 600) / 300) * 100));
  return (
    <div className="timeline">
      <span>T</span>
      <div className="bar">
        <u style={{ left: `${(600 / 900) * 100}%`, right: 0 }} />
        <i style={{ left: `${(600 / 900) * 100 + (position * 300) / 900}%` }} />
      </div>
      <span>+15 мин</span>
    </div>
  );
}

export function VehicleCard({ detail, alerts, policy, onAcknowledge, readOnly = false }: Props) {
  if (!detail) {
    return (
      <section className="card">
        <div className="empty">
          Выберите ТС в очереди или на карте, чтобы увидеть прогноз, цель и основания.
        </div>
      </section>
    );
  }
  const prediction = detail.prediction;
  const ok = prediction?.status === "ok";
  const previous = prediction?.status === "ml_unavailable" && prediction.delay_s != null;
  const available = ok || previous;
  const risk = available ? prediction!.risk_level ?? "green" : null;
  const fresh = freshness(detail);
  const alert = alerts.find(
    (item) => item.tr_id === detail.tr_id && item.state === "active" &&
      item.target_stop_id === prediction?.target_stop_id && prediction?.run_id === detail.run_id,
  );
  return (
    <section className="card" aria-label={`Карточка ТС ${detail.tr_id}`}>
      <h2>ТС {detail.tr_id}</h2>
      <div className="hint card-meta">
        <span>Устройство {detail.unit_id ?? "не сопоставлено"}</span>
        <span>{detail.events_in_window} событий</span>
        <span className={`badge ${fresh.tone}`}>{fresh.label}</span>
      </div>

      <CurrentDeviationCard value={detail.current_deviation} />

      {available ? (
        <>
          {previous ? <div className="banner warn">Последний прогноз · ML недоступен</div> : null}
          <div className="hint">Прогноз к целевой остановке · возраст: {duration(prediction!.prediction_age_s)}</div>
          <p className={`big risk-${risk}`}>{signedDelay(prediction!.delay_s)}</p>
          <div className="hint">
            {RISK_LABEL[risk!]} · {prediction!.risk_basis} · порог красного{" "}
            {policy?.red_min_delay_s ?? 120} с
          </div>
          <div className="section">
            <h3>Целевое посещение</h3>
            <div className="kv">
              <span className="k">Плановое прибытие</span>
              <span className="v">{sourceTime(prediction!.target_planned_at)}</span>
              <span className="k">Ожидаемое прибытие</span>
              <span className="v">{sourceTime(prediction!.expected_arrival_at)}</span>
              <span className="k">Горизонт прогноза</span>
              <span className="v">{duration(prediction!.horizon_s)}</span>
              <span className="k">Момент расчёта T</span>
              <span className="v">{sourceTime(prediction!.cutoff_t)}</span>
              <span className="k">Остановка</span>
              <span className="v">{prediction!.target_stop_id}</span>
            </div>
            <div className="hint" style={{ marginTop: 4 }}>
              {prediction!.target_address ?? "адрес остановки не указан"}
            </div>
            <div style={{ marginTop: 6 }}>
              <Timeline horizon={prediction!.horizon_s ?? 600} />
            </div>
          </div>

          <div className="section">
            <h3>Участок подхода к цели</h3>
            {detail.segment ? <>
              <div className="segment-stops">
                <div><span className="segment-label">От</span><span>{detail.segment.from.address ?? detail.segment.from.target_stop_id}</span></div>
                <div><span className="segment-label">До</span><span>{detail.segment.to.address ?? detail.segment.to.target_stop_id}</span></div>
              </div>
              <div className="hint">Схема двух последовательных плановых посещений; цвет линии соответствует риску ТС.</div>
            </> : <div className="hint">Участок не определён: нет предыдущего посещения либо разрыв плана больше 30 минут.</div>}
          </div>
          <div className="section">
            <h3>Риск опоздания</h3>
            <div className="kv">
              <span className="k">
                P(задержка &gt; {number(prediction!.late_threshold_s ?? 120, 0)} с)
              </span>
              <span className="v">{percent(prediction!.late_probability, 0)}</span>
              <span className="k">Модель</span>
              <span className="v">
                {prediction!.model_used === "fallback" ? "без подсказки" : "основная"}
              </span>
              <span className="k">Подсказка cur_dev</span>
              <span className="v">
                {signedDelay(prediction!.cur_dev_s)}
                {prediction!.cur_dev_age_s ? ` (${duration(prediction!.cur_dev_age_s)} назад)` : ""}
              </span>
              <span className="k">Источник подсказки</span>
              <span className="v">
                {HINT_LABEL[prediction!.cur_dev_source ?? ""] ?? prediction!.cur_dev_source ?? "—"}
              </span>
            </div>
            <div className="hint" style={{ marginTop: 4 }}>
              {CALIBRATION_NOTE[prediction!.calibration?.status ?? "unavailable"]}
              {prediction!.calibration?.report ? ` · ${prediction!.calibration.report}` : ""}
            </div>
          </div>

          <div className="section">
            <h3>Наблюдаемые основания</h3>
            {prediction!.evidence.length ? (
              <ul className="plain">
                {prediction!.evidence.map((item) => (
                  <li key={item.kind}>{item.text}</li>
                ))}
              </ul>
            ) : (
              <div className="hint">Особых отклонений в признаках не зафиксировано.</div>
            )}
            <div className="hint" style={{ marginTop: 4 }}>
              Это наблюдаемые закономерности, а не подтверждённая причина: данных о ДТП и дверях
              в источнике нет.
            </div>
          </div>

          <div className="section">
            <h3>Предлагаемое действие</h3>
            <div className="advice">{prediction!.recommendation}</div>
            {alert && ok ? (
              <div className="ack-actions">
                <button onClick={() => onAcknowledge(alert.alert_id)} disabled={readOnly || !!alert.acknowledged_at}>
                  {alert.acknowledged_at ? "Принято в работу" : readOnly ? "История · только просмотр" : "Отметить «принято в работу»"}
                </button>
                <span className="hint">
                  алерт с {sourceTime(alert.first_alert_at)} · обновлений {alert.updates}
                </span>
              </div>
            ) : null}
          </div>
        </>
      ) : (
        <>
          <p className="big risk-none">нет прогноза</p>
          <div className="hint">
            {STATUS_LABEL[prediction?.status ?? ""] ?? "Прогноз недоступен"}
            {prediction?.detail ? ` · ${prediction.detail}` : ""}
          </div>
          <div className="section">
            <h3>Почему</h3>
            <div className="hint">
              Отсутствие прогноза не равно нулевой задержке. Для расчёта нужны плановая остановка
              в окне 10–15 минут, свежая телеметрия и доступный ML-сервис.
            </div>
          </div>
        </>
      )}

      <div className="section">
        <h3>Качество данных</h3>
        <div className="kv">
          <span className="k">Последнее событие</span>
          <span className="v">{sourceTime(detail.last_event_at)}</span>
          <span className="k">Возраст телеметрии</span>
          <span className="v">{duration(detail.telemetry_age_s)}</span>
          <span className="k">Возраст позиции</span>
          <span className="v">{duration(detail.position_age_s)}</span>
          <span className="k">Невалидных координат</span>
          <span className="v">{percent(detail.invalid_gps_fraction, 0)}</span>
          <span className="k">Скорость</span>
          <span className="v">
            {detail.position?.speed_kmh !== null && detail.position?.speed_kmh !== undefined
              ? `${number(detail.position.speed_kmh, 0)} км/ч`
              : "—"}
          </span>
        </div>
        {detail.quality_flags.length ? (
          <div className="hint" style={{ marginTop: 4 }}>
            флаги: {detail.quality_flags.join(", ")}
          </div>
        ) : null}
      </div>

      {detail.prediction_history.length > 1 ? (
        <div className="section">
          <h3>История прогнозов</h3>
          <table className="prediction-history">
            <tbody>
              {detail.prediction_history
                .slice()
                .reverse()
                .slice(0, 8)
                .map((item) => (
                  <tr key={item.prediction_id}>
                    <td style={{ color: "var(--muted)" }}>{sourceTime(item.cutoff_t)}</td>
                    <td className={`risk-${item.risk_level ?? "none"}`} style={{ textAlign: "right" }}>
                      {signedDelay(item.delay_s)}
                    </td>
                    <td style={{ textAlign: "right", color: "var(--muted)" }}>
                      {percent(item.late_probability, 0)}
                    </td>
                    <td style={{ textAlign: "right", color: "var(--muted)" }}>
                      {item.target_stop_id?.slice(-4) ?? "—"}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          <div className="hint">Последние колонки — вероятность опоздания и хвост ID цели.</div>
        </div>
      ) : null}
    </section>
  );
}
