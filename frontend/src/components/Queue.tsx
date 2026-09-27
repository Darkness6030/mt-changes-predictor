import { useMemo } from "react";
import {
  RISK_LABEL,
  RISK_SIGN,
  STATUS_LABEL,
  freshness,
  severity,
  tripEdge,
  signedDelay,
  sourceTime,
  stopLabel,
} from "../format";
import type { Snapshot, Vehicle } from "../types";

export type Filter = "attention" | "all" | "late" | "early" | "nodata";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "attention", label: "Требуют внимания" },
  { key: "all", label: "Все" },
  { key: "late", label: "Опоздание" },
  { key: "early", label: "Опережение" },
  { key: "nodata", label: "Нет прогноза" },
];

interface Props {
  snapshot: Snapshot | null;
  filter: Filter;
  search: string;
  selected: string | null;
  onFilter: (filter: Filter) => void;
  onSearch: (value: string) => void;
  onSelect: (trId: string) => void;
  onFocus: (trId: string) => void;
}

function matches(vehicle: Vehicle, filter: Filter): boolean {
  const prediction = vehicle.prediction;
  const ok = prediction?.status === "ok";
  const available = ok || (prediction?.status === "ml_unavailable" && prediction.delay_s != null);
  switch (filter) {
    case "all":
      return true;
    case "attention":
      return available && (prediction!.attention != null ||
        prediction!.risk_level === "red" || prediction!.risk_level === "yellow");
    case "late":
      return available && (prediction!.delay_s ?? 0) > 0;
    case "early":
      return available && (prediction!.delay_s ?? 0) < 0;
    case "nodata":
      return !ok;
  }
}

/** Работа диспетчера начинается здесь: сначала те, у кого прогноз хуже и цель ближе. */
export function Queue({ snapshot, filter, search, selected, onFilter, onSearch, onSelect, onFocus }: Props) {
  const policy = snapshot?.risk_policy ?? null;
  const rows = useMemo(() => {
    const vehicles = snapshot?.vehicles ?? [];
    const query = search.trim().toLowerCase();
    return vehicles
      .filter((vehicle) => matches(vehicle, filter))
      .filter(
        (vehicle) =>
          !query ||
          vehicle.tr_id.toLowerCase().includes(query) ||
          (vehicle.unit_id ?? "").toLowerCase().includes(query) ||
          (vehicle.prediction?.target_address ?? "").toLowerCase().includes(query),
      )
      .sort((left, right) => {
        const bySeverity = severity(right.prediction) - severity(left.prediction);
        if (bySeverity !== 0) return bySeverity;
        const leftTarget = left.prediction?.target_planned_at ?? "";
        const rightTarget = right.prediction?.target_planned_at ?? "";
        if (leftTarget && rightTarget && leftTarget !== rightTarget) {
          return leftTarget < rightTarget ? -1 : 1;
        }
        return left.tr_id.localeCompare(right.tr_id);
      });
  }, [snapshot, filter, search]);

  return (
    <aside className="queue">
      <div className="filters">
        <input
          type="search"
          value={search}
          placeholder="Поиск по ТС, устройству или адресу"
          aria-label="Поиск транспортного средства"
          onChange={(event) => onSearch(event.target.value)}
        />
        <div className="chiprow" role="group" aria-label="Фильтр очереди">
          {FILTERS.map((item) => (
            <button
              key={item.key}
              className={filter === item.key ? "active" : ""}
              aria-pressed={filter === item.key}
              onClick={() => onFilter(item.key)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
      {snapshot?.hotspots?.length ? (
        <div className="hotspots" aria-label="Где копится опоздание">
          <div className="hotspots-title">Где копится опоздание · за прогон</div>
          {snapshot.hotspots.slice(0, 3).map((spot) => (
            <button
              key={spot.segment_id}
              className="hotspot"
              title="Прирост отклонения между соседними остановками, наблюдаемый по GPS. Нажмите, чтобы открыть ТС"
              onClick={() => { onSelect(spot.tr_id); onFocus(spot.tr_id); }}
            >
              <span className="hotspot-gain risk-red">{signedDelay(spot.gain_total_s)}</span>
              <span className="hotspot-where">
                {stopLabel(spot.from.address, spot.from.target_stop_id)} →{" "}
                {stopLabel(spot.to.address, spot.to.target_stop_id)}
              </span>
              <span className="hotspot-meta">
                ТС {spot.tr_id} · проездов {spot.passes}
              </span>
            </button>
          ))}
        </div>
      ) : null}
      <div className="list">
        {rows.length === 0 ? (
          <div className="empty">Под фильтр ничего не подходит.</div>
        ) : (
          rows.map((vehicle) => {
            const prediction = vehicle.prediction;
            const ok = prediction?.status === "ok";
            const previous = prediction?.status === "ml_unavailable" && prediction.delay_s != null;
            const available = ok || previous;
            const risk = available ? prediction!.risk_level ?? "green" : null;
            const fresh = freshness(vehicle);
            return (
              <button
                key={vehicle.tr_id}
                className={`row${selected === vehicle.tr_id ? " selected" : ""}`}
                onClick={() => onSelect(vehicle.tr_id)}
                onDoubleClick={() => onFocus(vehicle.tr_id)}
                title="Двойное нажатие — центрировать маршрут"
                aria-current={selected === vehicle.tr_id}
              >
                <div className="top">
                  <span className={risk ? `risk-${risk}` : "risk-none"} aria-hidden="true">
                    {risk ? RISK_SIGN[risk] : "○"}
                  </span>
                  <span className="id">ТС {vehicle.tr_id}</span>
                  <span className={`delay ${risk ? `risk-${risk}` : "risk-none"}`}>
                    {available ? signedDelay(prediction!.delay_s) : "нет прогноза"}
                  </span>
                </div>
                {previous ? <div className="hint">Последний прогноз · {Math.round(prediction!.prediction_age_s ?? 0)} с назад</div> : null}
                <div className="meta">
                  <span>{ok ? RISK_LABEL[risk!] : STATUS_LABEL[prediction?.status ?? ""] ?? "—"}</span>
                  {ok && prediction!.target_planned_at ? (
                    <span className="target-time">цель {sourceTime(prediction!.target_planned_at)}</span>
                  ) : null}
                </div>
                <div className="row-badges">
                  <span className={`badge ${fresh.tone}`}>{fresh.label}</span>
                  {ok && prediction!.late_probability !== null ? (
                    <span
                      className={policy && prediction!.late_probability >= policy.late_probability_red
                        ? "badge stale" : "badge"}
                      title={`Вероятность задержки больше ${Math.round(prediction!.late_threshold_s ?? 120)} с`}
                    >
                      опоздание {Math.round(prediction!.late_probability * 100)}%
                    </span>
                  ) : null}
                  {ok && tripEdge(prediction) ? (
                    <span
                      className={prediction!.attention ? "badge stale" : "badge"}
                      title={`Рейс ${prediction!.trip!.number} из ${prediction!.trip!.total}: крайние рейсы дня критичны для выполнения плана`}
                    >
                      {tripEdge(prediction)}
                    </span>
                  ) : null}
                  {ok && prediction!.attention === "probability" ? (
                    <span className="badge stale" title="Задержка в норме, но опоздание вероятно">
                      по вероятности
                    </span>
                  ) : null}
                </div>
                {ok ? (
                  <div className="addr" title={prediction!.target_address ?? ""}>
                    {stopLabel(prediction!.target_address, prediction!.target_stop_id)}
                  </div>
                ) : null}
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
