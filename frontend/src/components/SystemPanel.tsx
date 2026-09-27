import { duration, number, sourceTime } from "../format";
import type { Quality, Status } from "../types";

interface Props {
  status: Status | null;
  quality: Quality | null;
}

function Row({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <tr title={title}>
      <td style={{ color: "var(--muted)" }}>{label}</td>
      <td>{value}</td>
    </tr>
  );
}

/** Technical evidence for the jury: versions, latency, counters and measured quality. */
export function SystemPanel({ status, quality }: Props) {
  const performance = status?.performance;
  const sidecar = quality?.replay_sidecar;
  const offline = quality?.offline;
  return (
    <section className="card system" aria-label="Система и измерения">
      <h2>Система · сейчас</h2>
      <div className="hint">
        Backend {status?.backend_version ?? "—"} · run {status?.run_id ?? "—"} · режим{" "}
        {status?.mode ?? "—"} · выборка {status?.split ?? "—"}
      </div>

      <div className="section">
        <h3>Модель</h3>
        <table>
          <tbody>
            <Row label="ML готов" value={status?.ml.ready ? "да" : "нет"} />
            <Row
              label="Версия модели"
              value={(status?.ml.model_version ?? "—").slice(0, 12)}
              title={status?.ml.model_version ?? ""}
            />
            <Row label="Калибровка" value={status?.ml.calibration?.status ?? "—"} />
            <Row label="Запросов к ML" value={String(status?.ml.requests ?? 0)} />
            <Row
              label="Ошибок / таймаутов"
              value={`${status?.ml.errors ?? 0} / ${status?.ml.timeouts ?? 0}`}
            />
          </tbody>
        </table>
        {status?.ml.last_error ? <div className="hint mono">{status.ml.last_error}</div> : null}
      </div>

      <div className="section">
        <h3>Задержки, мс</h3>
        <table>
          <tbody>
            <Row
              label="ML inference p50 / p95"
              value={`${number(performance?.ml_inference_ms.p50_ms, 1)} / ${number(performance?.ml_inference_ms.p95_ms, 1)}`}
            />
            <Row
              label="Обращение к ML p50 / p95"
              value={`${number(performance?.ml_round_trip_ms.p50_ms, 1)} / ${number(performance?.ml_round_trip_ms.p95_ms, 1)}`}
            />
            <Row
              label="Цикл Backend p50 / p95"
              value={`${number(performance?.cycle_ms.p50_ms, 1)} / ${number(performance?.cycle_ms.p95_ms, 1)}`}
            />
            <Row
              label="Сборка признаков p50"
              value={number(performance?.feature_build_ms.p50_ms, 1)}
            />
            <Row
              label="Событие → прогноз p50 / p95"
              value={`${number(performance?.event_to_prediction_ms.p50_ms, 0)} / ${number(performance?.event_to_prediction_ms.p95_ms, 0)}`}
              title="Настенное время от последнего принятого события до опубликованного прогноза"
            />
            <Row label="Прогнозов выдано" value={String(performance?.predictions ?? 0)} />
            <Row
              label="Период опроса / прогноза, с"
              value={`${performance?.tick_interval_s ?? "—"} / ${performance?.predict_interval_s ?? "—"}`}
            />
          </tbody>
        </table>
        <div className="hint">
          Опрос UI раз в секунду входит в полную задержку доставки и учитывается отдельно.
        </div>
      </div>

      <div className="section">
        <h3>Поток</h3>
        <table>
          <tbody>
            {status?.ndtp ? (
              <>
                <Row label="Соединений всего / открыто" value={`${status.ndtp.connections_total} / ${status.ndtp.connections_open}`} />
                <Row label="Handshake / realtime" value={`${status.ndtp.handshakes} / ${status.ndtp.realtime_frames}`} />
                <Row label="Навигационных ячеек" value={String(status.ndtp.nav_events)} />
                <Row label="Ошибок CRC / кадров" value={`${status.ndtp.crc_errors} / ${status.ndtp.invalid_frames}`} />
                <Row label="Неизвестных ячеек" value={String(status.ndtp.unknown_cells)} />
                <Row label="Обрывов / таймаутов" value={`${status.ndtp.disconnects} / ${status.ndtp.read_timeouts}`} />
                <Row label="Последний кадр" value={String(status.ndtp.last_frame_at ?? "—")} />
              </>
            ) : null}
            {status?.replay ? (
              <>
                <Row
                  label="События replay"
                  value={`${status.replay.events_delivered} / ${status.replay.events_total}`}
                />
                <Row label="Период пройден" value={status.replay.finished ? "полностью" : "идёт"} />
              </>
            ) : null}
            {status?.points ? (
              <Row
                label="Точек: всего / посчитано / пропущено"
                value={`${status.points.total} / ${status.points.predicted} / ${status.points.skipped}`}
              />
            ) : null}
            <Row label="ТС в состоянии" value={String(status?.state.vehicles ?? "—")} />
            <Row label="Событий в памяти" value={String(status?.state.events_in_state ?? "—")} />
            <Row label="Дубликатов" value={String(status?.state.duplicate_events ?? "—")} />
            <Row label="Скрыто GPS-точек" value={String(status?.state.suspect_gps_fixes ?? "—")} />
            <Row label="Событий из будущего" value={String(status?.state.rejected_future_events ?? "—")} />
            <Row label="Несопоставленных устройств" value={String(status?.state.unmapped_units ?? "—")} />
          </tbody>
        </table>
      </div>

      {status?.history?.enabled ? <div className="section">
        <h3>Журнал просмотра</h3>
        <table><tbody>
          <Row label="Записанных снимков" value={String(status.history.frames)} />
          <Row label="Память / предел, МиБ" value={`${number(status.history.bytes / 1048576, 1)} / ${number(status.history.max_bytes / 1048576, 0)}`} />
          <Row label="Запись p50 / p95, мс" value={`${number(performance?.history_capture_ms?.p50_ms, 1)} / ${number(performance?.history_capture_ms?.p95_ms, 1)}`} />
          <Row label="Удалено старых снимков" value={String(status.history.evicted_frames)} />
          <Row label="Пропущено: часы / размер" value={`${status.history.skipped_clock_samples} / ${status.history.oversized_frames}`} />
        </tbody></table>
        <div className="hint">До {Math.round(status.history.window_s / 60)} мин. времени источника; запись не чаще раза в {status.history.sample_interval_s} с реального времени. Новый прогон очищает журнал.</div>
      </div> : null}

      <div className="section">
        <h3>Измеренное качество</h3>
        {offline ? (
          <table>
            <tbody>
              <Row label="Offline MAE модели, с" value={number(offline.model_mae_s, 2)} />
              <Row label="Без подсказки, с" value={number(offline.no_hint_mae_s, 2)} />
              <Row label="Baseline cur_dev, с" value={number(offline.cur_dev_mae_s, 2)} />
              <Row label="Нулевой прогноз, с" value={number(offline.zero_mae_s, 2)} />
              <Row label="Строк" value={String(offline.rows)} />
            </tbody>
          </table>
        ) : (
          <div className="hint">Offline-отчёт не подключён.</div>
        )}
        {offline ? <div className="hint">{offline.note}</div> : null}
        {sidecar ? (
          <>
            <h3 style={{ marginTop: 10 }}>Этот прогон по разметке</h3>
            {sidecar.measured_rows ? (
              <table>
                <tbody>
                  <Row label="MAE потока, с" value={number(sidecar.mae_s, 2)} />
                  <Row label="Baseline cur_dev, с" value={number(sidecar.cur_dev_mae_s, 2)} />
                  <Row label="Медиана ошибки, с" value={number(sidecar.absolute_error_p50_s, 2)} />
                  <Row
                    label="Lead time min / p50"
                    value={`${duration(sidecar.lead_time_s?.min)} / ${duration(sidecar.lead_time_s?.p50)}`}
                  />
                  <Row label="Brier вероятности" value={number(sidecar.late_probability_brier, 3)} />
                  <Row
                    label="Точек измерено / ждут события"
                    value={`${sidecar.measured_rows} / ${sidecar.pending_rows}`}
                  />
                </tbody>
              </table>
            ) : (
              <div className="hint">{sidecar.note ?? "Ожидание наступления целевых времён."}</div>
            )}
            <div className="hint">
              Метрика считается только после фактического времени цели; разметка не участвует в
              инференсе.
            </div>
          </>
        ) : null}
      </div>

      <div className="section">
        <h3>Ссылки и часы</h3>
        <div className="hint">
          <a href="/docs" target="_blank" rel="noreferrer">
            Swagger Backend
          </a>{" "}
          ·{" "}
          <a href="/openapi.json" target="_blank" rel="noreferrer">
            OpenAPI
          </a>
        </div>
        <table>
          <tbody>
            <Row label="Время источника" value={sourceTime(status?.clock.source_time)} />
            <Row label="Настенное время" value={status?.clock.wall_time ?? "—"} />
            <Row label="Сдвиг плана, с" value={String(status?.clock.plan_shift_s ?? 0)} />
            <Row label="Базис времени" value={status?.clock.time_basis ?? "—"} />
          </tbody>
        </table>
        {Object.keys(status?.errors ?? {}).length ? (
          <div className="hint mono">
            Ошибки цикла: {JSON.stringify(status?.errors)}
          </div>
        ) : null}
      </div>
    </section>
  );
}
