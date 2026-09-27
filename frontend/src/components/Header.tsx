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
      <div className="kpis" aria-label="Сводка для диспетчера">
        <div className="kpi red" title="ТС с риском по задержке или по вероятности опоздания">
          <b>{summary?.attention ?? "—"}</b>
          <span>требуют внимания</span>
        </div>
        <div
          className={`kpi ${(summary?.edge_trips_at_risk ?? 0) > 0 ? "red" : "grey"}`}
          title="Первый или последний рейс дня под угрозой опоздания"
        >
          <b>{summary?.edge_trips_at_risk ?? 0}</b>
          <span>крайние рейсы в риске</span>
        </div>
        <div className="kpi" title="Открытые алерты раннего предупреждения">
          <b>{summary?.alerts_active ?? "—"}</b>
          <span>активных алертов</span>
        </div>
        <div className="kpi">
          <b>
            {summary?.with_prediction ?? "—"}
            <small> / {summary?.vehicles ?? "—"}</small>
          </b>
          <span>{historical ? "с прогнозом в снимке" : "ТС с прогнозом"}</span>
        </div>
        <div className="kpi yellow" title="Позиция старше порога: показано последнее состояние">
          <b>{summary?.stale ?? "—"}</b>
          <span>устаревшие данные</span>
        </div>
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
