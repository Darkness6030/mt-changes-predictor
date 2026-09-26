import { duration, sourceDate, sourceTime } from "../format";
import type { Snapshot, Status } from "../types";

interface Props {
  snapshot: Snapshot | null;
  status: Status | null;
  ageMs: number | null;
}

/** Dispatcher summary; replay clock and speed are shown in the navigation panel. */
export function Header({ snapshot, status, ageMs }: Props) {
  const clock = snapshot?.clock ?? status?.clock ?? null;
  const mode = snapshot?.mode ?? status?.mode ?? "—";
  const modeLabel = mode === "ndtp" ? "Живой NDTP" : "Исторический replay";
  const summary = snapshot?.summary;
  return (
    <header className="header">
      <div className="header-info">
        <h1>Пульт диспетчера · прогноз задержек</h1>
        <div className="sub">
          {mode === "replay" ? (
            clock?.paused ? <span className="replay-paused" role="status">Воспроизведение на паузе</span> : null
          ) : (
            <>
              <span>{modeLabel}</span>
              <span title="Время источника; часовой пояс в датасете не установлен">
                <b>{sourceTime(clock?.source_time)}</b> {sourceDate(clock?.source_time)} · зона не задана
              </span>
            </>
          )}
          {ageMs !== null ? <span>Обновлено {Math.round(ageMs / 1000)} с назад</span> : null}
        </div>
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
          <span>ТС в потоке</span>
        </div>
        {status?.performance ? (
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
        {status?.points ? (
          <div className="kpi" title="Официальные прогнозные точки, обработанные в потоке">
            <b>{status.points.predicted}</b>
            <span>точек посчитано</span>
          </div>
        ) : null}
        {status?.ndtp ? (
          <div className="kpi" title="Принятые NDTP-пакеты телематики">
            <b>{String(status.ndtp.realtime_frames ?? 0)}</b>
            <span>NDTP-пакетов</span>
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
