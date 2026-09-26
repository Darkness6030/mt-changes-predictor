import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { signedDelay, sourceTime } from "../format";
import { forecastColour as colourOf, RISK_COLOURS as COLOURS } from "../mapPresentation";
import { VehicleMarker } from "../vehicleMarker";
import type { Snapshot, VehicleDetail } from "../types";

const MOSCOW: L.LatLngTuple = [55.751244, 37.618423];
const TILE_URL =
  (import.meta.env.VITE_TILE_URL as string | undefined) ??
  "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
interface Props {
  snapshot: Snapshot | null;
  detail: VehicleDetail | null;
  selected: string | null;
  onSelect: (trId: string) => void;
  focusRequest: { trId: string; sequence: number } | null;
}

/**
 * Leaflet map with vehicles, the selected plan sequence and the already observed track.
 * Nothing here is a road route: straight lines between planned stops are a schematic.
 */
export function MapView({ snapshot, detail, selected, onSelect, focusRequest }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const markers = useRef<Map<string, VehicleMarker>>(new Map());
  const overlay = useRef<L.LayerGroup | null>(null);
  const paths = useRef<Map<string, L.Polyline | L.CircleMarker>>(new Map());
  const focused = useRef<{ sequence: number; fullPlan: boolean } | null>(null);
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
      paths.current.clear();
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
      let marker = markers.current.get(vehicle.tr_id);
      if (!marker) {
        marker = new VehicleMarker(vehicle, instance, (trId) => select.current(trId));
        markers.current.set(vehicle.tr_id, marker);
      }
      marker.update(vehicle, vehicle.tr_id === selected);
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
    // Reconcile by stable identity: polling must not remove the layer under the pointer.
    const seen = new Set<string>();
    let topologyChanged = false;
    const setTooltip = (layer: L.Polyline | L.CircleMarker, text?: string) => {
      if (!text) return;
      const existing = layer.getTooltip()?.getContent();
      if (existing instanceof HTMLElement && existing.textContent === text) return;
      const label = document.createElement("span");
      label.textContent = text;
      if (layer.getTooltip()) layer.setTooltipContent(label);
      else layer.bindTooltip(label, { direction: "top" });
    };
    const line = (key: string, points: L.LatLngTuple[], style: L.PolylineOptions, text?: string, trId?: string) => {
      seen.add(key);
      let layer = paths.current.get(key) as L.Polyline | undefined;
      if (!layer) {
        layer = L.polyline(points, style).addTo(group);
        if (trId) layer.on("click", () => select.current(trId));
        paths.current.set(key, layer);
        topologyChanged = true;
      } else { layer.setLatLngs(points); layer.setStyle(style); }
      setTooltip(layer, text);
    };
    const point = (key: string, position: L.LatLngTuple, style: L.CircleMarkerOptions, text: string) => {
      seen.add(key);
      let layer = paths.current.get(key) as L.CircleMarker | undefined;
      if (!layer) {
        layer = L.circleMarker(position, style).addTo(group);
        paths.current.set(key, layer);
        topologyChanged = true;
      } else { layer.setLatLng(position); layer.setStyle(style); layer.setRadius(style.radius ?? 4); }
      setTooltip(layer, text);
    };
    for (const vehicle of vehicles) {
      const segment = vehicle.segment;
      if (!segment) continue;
      line(`segment:${vehicle.tr_id}`, [[segment.from.lat, segment.from.lon], [segment.to.lat, segment.to.lon]], {
        color: colourOf(vehicle), weight: vehicle.tr_id === selected ? 7 : 4,
        opacity: vehicle.stale ? 0.35 : 0.8,
        dashArray: vehicle.prediction?.status === "ok" ? undefined : "5 5",
      }, `ТС ${vehicle.tr_id}: ${segment.from.address ?? "посещение"} → ${segment.to.address ?? "цель"} (схема)`, vehicle.tr_id);
    }
    if (detail) {
      const plan = detail.plan ?? [];
      if (plan.length > 1) line(`plan:${detail.tr_id}`,
        plan.map((visit) => [visit.lat, visit.lon]),
        { color: "#4da3ff", weight: 2, opacity: 0.5, dashArray: "6 6" });
      for (const visit of plan) point(`visit:${detail.tr_id}:${visit.target_stop_id}`, [visit.lat, visit.lon], {
        radius: 4, color: visit.passed ? "#4a5a6d" : "#4da3ff",
        fillColor: visit.passed ? "#2a3542" : "#1d3a58", fillOpacity: 1, weight: 1,
      }, `${sourceTime(visit.planned_at)} · ${visit.address ?? "адрес не указан"}` + (visit.passed ? " · пройдено" : ""));
      const track = (detail.track ?? []).filter((item) => item.lon !== null && item.lat !== null);
      if (track.length > 1) line(`track:${detail.tr_id}`,
        track.map((item) => [item.lat as number, item.lon as number]),
        { color: "#e8eef5", weight: 2, opacity: 0.8 });
      const prediction = detail.prediction;
      if (prediction?.status === "ok" && prediction.target_lat != null && prediction.target_lon != null) {
        point(`target:${detail.tr_id}:${prediction.target_stop_id}`, [prediction.target_lat, prediction.target_lon], {
          radius: 9, color: "#ffffff", weight: 2,
          fillColor: COLOURS[prediction.risk_level ?? "green"], fillOpacity: 0.85,
        }, `Целевая остановка · план ${sourceTime(prediction.target_planned_at)} · ${signedDelay(prediction.delay_s)}`);
      }
    }
    for (const [key, layer] of paths.current) {
      if (!seen.has(key)) { layer.remove(); paths.current.delete(key); topologyChanged = true; }
    }
    // Stops and schematic lines must not capture hover/click above the bus at the same GPS.
    if (topologyChanged) for (const marker of markers.current.values()) marker.bringToFront();
  }, [detail, vehicles, selected]);

  const focus = () => {
    const instance = map.current;
    if (!instance || !selected) return;
    const vehicle = vehicles.find((item) => item.tr_id === selected);
    const points: L.LatLngTuple[] = detail?.tr_id === selected
      ? (detail.plan ?? []).map((visit) => [visit.lat, visit.lon]) : [];
    if (vehicle?.segment) points.push([vehicle.segment.from.lat, vehicle.segment.from.lon],
      [vehicle.segment.to.lat, vehicle.segment.to.lon]);
    const marker = markers.current.get(selected);
    if (marker) points.push([marker.getLatLng().lat, marker.getLatLng().lng]);
    if (points.length) instance.fitBounds(L.latLngBounds(points), { padding: [30, 30], maxZoom: 15 });
  };

  useEffect(() => {
    if (!focusRequest || focusRequest.trId !== selected || !vehicles.some((item) => item.tr_id === selected)) return;
    const fullPlan = detail?.tr_id === selected && (detail.plan?.length ?? 0) > 0;
    if (focused.current?.sequence === focusRequest.sequence && (focused.current.fullPlan || !fullPlan)) return;
    focus();
    focused.current = { sequence: focusRequest.sequence, fullPlan };
  }, [focusRequest, selected, detail, vehicles]);

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
          <span className="vehicle-colour-key" aria-hidden="true" />
          <span>Обводка ТС — текущее отклонение по GPS; заливка и линия — прогноз к цели.</span>
        </div>
        <div className="legend-note">Серая пунктирная обводка — оценка отсутствует или устарела.
          Белый внешний ореол — выбранное ТС. Текущее отклонение оценено на последней
          распознанной остановке, возраст указан в подсказке.</div>
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
