import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { RISK_LABEL, signedDelay, sourceTime } from "../format";
import type { Snapshot, Vehicle, VehicleDetail } from "../types";

const MOSCOW: L.LatLngTuple = [55.751244, 37.618423];
const TILE_URL =
  (import.meta.env.VITE_TILE_URL as string | undefined) ??
  "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const COLOURS: Record<string, string> = {
  green: "#2fbf71",
  yellow: "#f2b63c",
  red: "#ff5d5d",
  none: "#6b7a8d",
};

function colourOf(vehicle: Vehicle): string {
  const prediction = vehicle.prediction;
  if (!prediction || !["ok", "ml_unavailable"].includes(prediction.status) || !prediction.risk_level) return COLOURS.none;
  return COLOURS[prediction.risk_level];
}

interface Props {
  snapshot: Snapshot | null;
  detail: VehicleDetail | null;
  selected: string | null;
  onSelect: (trId: string) => void;
}

/**
 * Leaflet map with vehicles, the selected plan sequence and the already observed track.
 * Nothing here is a road route: straight lines between planned stops are a schematic.
 */
export function MapView({ snapshot, detail, selected, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const markers = useRef<Map<string, L.CircleMarker>>(new Map());
  const overlay = useRef<L.LayerGroup | null>(null);
  const tiles = useRef<L.TileLayer | null>(null);
  const [tilesFailed, setTilesFailed] = useState(false);
  const [basemap, setBasemap] = useState(true);
  const select = useRef(onSelect);
  select.current = onSelect;

  useEffect(() => {
    if (map.current || !container.current) return;
    const instance = L.map(container.current, {
      center: MOSCOW,
      zoom: 10,
      zoomControl: true,
      preferCanvas: true,
      attributionControl: true,
    });
    overlay.current = L.layerGroup().addTo(instance);
    map.current = instance;
    // The panel also changes size at layout breakpoints, without a window resize.
    const resize = new ResizeObserver(() => instance.invalidateSize({ pan: false }));
    resize.observe(container.current);
    return () => {
      resize.disconnect();
      instance.remove();
      map.current = null;
      markers.current.clear();
    };
  }, []);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    if (!basemap) {
      tiles.current?.remove();
      tiles.current = null;
      return;
    }
    const layer = L.tileLayer(TILE_URL, {
      maxZoom: 18,
      attribution: "© OpenStreetMap contributors",
    });
    layer.on("tileerror", () => setTilesFailed(true));
    layer.addTo(instance);
    tiles.current = layer;
    return () => {
      layer.remove();
      tiles.current = null;
    };
  }, [basemap]);

  const vehicles = useMemo(() => snapshot?.vehicles ?? [], [snapshot]);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const seen = new Set<string>();
    for (const vehicle of vehicles) {
      const position = vehicle.position;
      if (!position || position.lon === null || position.lat === null) continue;
      seen.add(vehicle.tr_id);
      const latlng: L.LatLngTuple = [position.lat, position.lon];
      const isSelected = vehicle.tr_id === selected;
      const outdated = vehicle.stale || vehicle.prediction?.status === "ml_unavailable";
      const style = {
        color: isSelected ? "#ffffff" : colourOf(vehicle),
        weight: isSelected ? 3 : 1.5,
        fillColor: colourOf(vehicle),
        fillOpacity: outdated ? 0.35 : 0.9,
        radius: isSelected ? 10 : 7,
        dashArray: outdated ? "3 3" : undefined,
      };
      const prediction = vehicle.prediction;
      const tooltip =
        `ТС ${vehicle.tr_id}` +
        (prediction?.status === "ok"
          ? ` · ${signedDelay(prediction.delay_s)} · ${RISK_LABEL[prediction.risk_level ?? "green"]}` +
            ` · цель ${sourceTime(prediction.target_planned_at)}`
          : " · нет прогноза") +
        (vehicle.stale ? " · данные устарели" : "");
      const existing = markers.current.get(vehicle.tr_id);
      if (existing) {
        existing.setLatLng(latlng);
        existing.setStyle(style);
        existing.setRadius(style.radius);
        existing.setTooltipContent(tooltip);
      } else {
        const marker = L.circleMarker(latlng, style)
          .bindTooltip(tooltip, { direction: "top" })
          .on("click", () => select.current(vehicle.tr_id))
          .addTo(instance);
        markers.current.set(vehicle.tr_id, marker);
      }
    }
    for (const [trId, marker] of markers.current) {
      if (!seen.has(trId)) {
        marker.remove();
        markers.current.delete(trId);
      }
    }
  }, [vehicles, selected]);

  useEffect(() => {
    const group = overlay.current;
    if (!group) return;
    group.clearLayers();
    for (const vehicle of vehicles) {
      const segment = vehicle.segment;
      if (!segment) continue;
      const line = L.polyline([[segment.from.lat, segment.from.lon], [segment.to.lat, segment.to.lon]], {
        color: colourOf(vehicle), weight: vehicle.tr_id === selected ? 7 : 4,
        opacity: vehicle.stale ? 0.35 : 0.8,
        dashArray: vehicle.prediction?.status === "ok" ? undefined : "5 5",
      });
      const label = document.createElement("span");
      label.textContent = `ТС ${vehicle.tr_id}: ${segment.from.address ?? "посещение"} → ${segment.to.address ?? "цель"} (схема)`;
      line.bindTooltip(label).on("click", () => select.current(vehicle.tr_id)).addTo(group);
    }
    if (!detail) return;
    const plan = detail.plan ?? [];
    if (plan.length > 1) {
      L.polyline(
        plan.map((visit) => [visit.lat, visit.lon] as L.LatLngTuple),
        { color: "#4da3ff", weight: 2, opacity: 0.5, dashArray: "6 6" },
      ).addTo(group);
    }
    for (const visit of plan) {
      L.circleMarker([visit.lat, visit.lon], {
        radius: 4,
        color: visit.passed ? "#4a5a6d" : "#4da3ff",
        fillColor: visit.passed ? "#2a3542" : "#1d3a58",
        fillOpacity: 1,
        weight: 1,
      })
        .bindTooltip(
          `${sourceTime(visit.planned_at)} · ${visit.address ?? "адрес не указан"}` +
            (visit.passed ? " · пройдено" : ""),
          { direction: "top" },
        )
        .addTo(group);
    }
    const track = (detail.track ?? []).filter(
      (point) => point.lon !== null && point.lat !== null,
    ) as { lon: number; lat: number }[];
    if (track.length > 1) {
      L.polyline(
        track.map((point) => [point.lat, point.lon] as L.LatLngTuple),
        { color: "#e8eef5", weight: 2, opacity: 0.8 },
      ).addTo(group);
    }
    const prediction = detail.prediction;
    if (prediction?.status === "ok" && prediction.target_lat && prediction.target_lon) {
      L.circleMarker([prediction.target_lat, prediction.target_lon], {
        radius: 9,
        color: "#ffffff",
        weight: 2,
        fillColor: COLOURS[prediction.risk_level ?? "green"],
        fillOpacity: 0.85,
      })
        .bindTooltip(
          `Целевая остановка · план ${sourceTime(prediction.target_planned_at)} · ` +
            `${signedDelay(prediction.delay_s)}`,
          { direction: "top", permanent: false },
        )
        .addTo(group);
    }
  }, [detail, vehicles, selected]);

  const focus = () => {
    const instance = map.current;
    const marker = selected ? markers.current.get(selected) : null;
    if (instance && marker) instance.setView(marker.getLatLng(), Math.max(instance.getZoom(), 13));
  };

  return (
    <div className="map-wrap">
      <div id="map" ref={container} role="application" aria-label="Карта транспортных средств" />
      <div className="map-buttons">
        <button onClick={focus} disabled={!selected}>
          Центрировать выбранное
        </button>
        <button onClick={() => setBasemap((value) => !value)}>
          {basemap ? "Схема без подложки" : "Включить подложку"}
        </button>
      </div>
      {tilesFailed && basemap ? (
        <div className="map-note">
          Тайлы подложки недоступны. Список, геометрия и прогнозы работают; можно переключиться
          на схему без подложки.
        </div>
      ) : null}
      <details className="legend">
        <summary>Обозначения карты</summary>
        <div className="legend-content">
        <div className="item">
          <span className="risk-red" aria-hidden="true">
            ■
          </span>
          <span>Опоздание больше {snapshot?.risk_policy.red_min_delay_s ?? 120} с</span>
        </div>
        <div className="item">
          <span className="risk-yellow" aria-hidden="true">
            ▲
          </span>
          <span>Внимание: отклонение выше допустимого</span>
        </div>
        <div className="item">
          <span className="risk-green" aria-hidden="true">
            ●
          </span>
          <span>В графике</span>
        </div>
        <div className="item">
          <span className="risk-none" aria-hidden="true">
            ○
          </span>
          <span>Нет прогноза / данные устарели (пунктир)</span>
        </div>
        <div className="legend-note">
          Линии показывают риск на участке подхода к цели. Геометрия схематичная.
        </div>
        </div>
      </details>
    </div>
  );
}
