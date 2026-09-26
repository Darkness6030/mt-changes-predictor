import { duration } from "../format";
import type { Snapshot, Status } from "../types";

interface Props {
  historical?: boolean;
  snapshot: Snapshot | null;
  status: Status | null;
  ageMs: number | null;
  error: string | null;
}

/** Dispatcher summary; replay clock and speed are shown in the navigation panel. */
export function Header({ snapshot, status, ageMs, error, historical = false }: Props) {
  const clock = snapshot?.clock ?? status?.clock ?? null;
  const mode = snapshot?.mode ?? status?.mode ?? "—";
  const summary = snapshot?.summary;
  return (
    <header className="header">
      <div className="header-info">
        <h1>Пульт диспетчера · прогноз задержек</h1>
        <div className="sub">
          {mode === "replay" ? (
            clock?.paused ? <span className="replay-paused" role="status">Воспроизведение на паузе</span> : null
          ) : (
            <span title={historical ? "Просмотр истории; приём NDTP и новые прогнозы продолжаются" : undefined}>
              {historical ? "История · приём идёт" : "Живой NDTP"}
            </span>
          )}
          {ageMs !== null ? <span>{historical ? "Связь" : "Обновлено"} {Math.round(ageMs / 1000)} с назад</span> : null}
        </div>
        <div className="header-data-status" role={error ? "alert" : undefined} title={error ?? undefined}>{error}</div>
      </div>
      <div className="kpis">
        <div className="kpi">
          <b>{summary?.with_prediction ?? "—"}</b>
          <span>с прогнозом</span>
        </div>
        <div className="kpi red">
          <b>{summary?.attention ?? "—"}</b>
          <span>требуют внимания</span>
        </div>
        <div className="kpi grey">
          <b>{summary?.no_prediction ?? "—"}</b>
          <span>без прогноза</span>
        </div>
        <div className="kpi yellow">
          <b>{summary?.stale ?? "—"}</b>
          <span>устаревшие</span>
        </div>
        <div className="kpi">
          <b>{summary?.vehicles ?? "—"}</b>
          <span>{historical ? "ТС в снимке" : "ТС в потоке"}</span>
        </div>
        {!historical && status?.performance ? (
          <div className="kpi" title="p95 полного обращения Backend → ML → Backend">
            <b>{status.performance.ml_round_trip_ms.p95_ms?.toFixed(0) ?? "—"}</b>
            <span>p95 ML, мс</span>
          </div>
        ) : null}
        {clock?.source_progress !== undefined ? (
          <div className="kpi" title="Доставлено событий из всего периода данных">
            <b>{Math.round((clock.source_progress ?? 0) * 100)}%</b>
            <span>период пройден</span>
          </div>
        ) : null}
        {!historical && status?.points ? (
          <div className="kpi" title="Официальные прогнозные точки, обработанные в потоке">
            <b>{status.points.predicted}</b>
            <span>точек посчитано</span>
          </div>
        ) : null}
        {status?.ndtp ? (
          <div className="kpi" title="Принятые NDTP-пакеты телематики">
            <b>{String(status.ndtp.realtime_frames ?? 0)}</b>
            <span>{historical ? "пакетов сейчас" : "NDTP-пакетов"}</span>
          </div>
        ) : null}
        {ageMs !== null && ageMs > 5000 ? (
          <div className="kpi yellow" title="Возраст последнего успешного ответа API">
            <b>{duration(ageMs / 1000)}</b>
            <span>нет обновлений</span>
          </div>
        ) : null}
      </div>
    </header>
  );
}
