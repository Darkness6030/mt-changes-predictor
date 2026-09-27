import { useEffect, useMemo, useRef, useState } from "react";
import { signedDelay, sourceTime, stopLabel } from "../format";
import { forecastColour as colourOf, RISK_COLOURS as COLOURS } from "../mapPresentation";
import { VehicleMarker } from "../vehicleMarker";
import { hintHtml, loadYandexMaps, svgImage } from "../yandexMaps";
import type { Snapshot, VehicleDetail } from "../types";

const MOSCOW = [55.751244, 37.618423];
interface Props {
  snapshot: Snapshot | null;
  detail: VehicleDetail | null;
  selected: string | null;
  onSelect: (trId: string) => void;
  focusRequest: { trId: string; sequence: number } | null;
}

/**
 * Yandex map with vehicles, the selected plan sequence and the already observed track.
 * Nothing here is a road route: straight lines between planned stops are a schematic.
 */
export function MapView({ snapshot, detail, selected, onSelect, focusRequest }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<ymaps.Map | null>(null);
  const markers = useRef<Map<string, VehicleMarker>>(new Map());
  const paths = useRef<Map<string, ymaps.Polyline | ymaps.Placemark>>(new Map());
  const pathVersions = useRef<Map<string, string>>(new Map());
  const focused = useRef<{ sequence: number; fullPlan: boolean } | null>(null);
  const [ready, setReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const select = useRef(onSelect);
  select.current = onSelect;

  useEffect(() => {
    let cancelled = false;
    let instance: ymaps.Map | null = null;
    let resize: ResizeObserver | null = null;
    setMapError(null);
    setReady(false);
    loadYandexMaps().then((api) => {
      if (cancelled || !container.current) return;
      instance = new api.Map(container.current, {
        center: MOSCOW, zoom: 10, controls: ["zoomControl"],
      }, { suppressMapOpenBlock: true, copyrightLogoVisible: false });
      map.current = instance;
      resize = new ResizeObserver(() => instance?.container.fitToViewport());
      resize.observe(container.current);
      setReady(true);
    }).catch((error: Error) => { if (!cancelled) setMapError(error.message); });
    return () => {
      cancelled = true;
      resize?.disconnect();
      instance?.destroy();
      map.current = null;
      markers.current.clear();
      paths.current.clear();
      pathVersions.current.clear();
      focused.current = null;
    };
  }, [attempt]);

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
  }, [vehicles, selected, ready]);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    // Stable objects keep hover hints open across snapshot polling.
    const seen = new Set<string>();
    const setTooltip = (layer: ymaps.Polyline | ymaps.Placemark, text?: string) => {
      if (text && String(layer.properties.get("hintContent", {})) !== hintHtml(text)) {
        layer.properties.set("hintContent", hintHtml(text));
      }
    };
    const line = (key: string, points: number[][], style: {
      color: string; weight: number; opacity: number; dashArray?: string;
    }, text?: string, trId?: string) => {
      seen.add(key);
      const options = { strokeColor: style.color, strokeWidth: style.weight,
        strokeOpacity: style.opacity, strokeStyle: style.dashArray ? "dash" : "solid", zIndex: 1 };
      const version = JSON.stringify([points, options]);
      let layer = paths.current.get(key) as ymaps.Polyline | undefined;
      if (!layer) {
        layer = new ymaps.Polyline(points, {}, options);
        instance.geoObjects.add(layer);
        if (trId) layer.events.add("click", () => select.current(trId));
        paths.current.set(key, layer);
      } else if (pathVersions.current.get(key) !== version) {
        layer.geometry!.setCoordinates(points); layer.options.set(options);
      }
      pathVersions.current.set(key, version);
      setTooltip(layer, text);
    };
    const point = (key: string, position: number[], style: {
      radius: number; color: string; fillColor: string; fillOpacity: number; weight: number;
    }, text: string) => {
      seen.add(key);
      const size = (style.radius + style.weight) * 2;
      const icon = svgImage(`<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}"><circle cx="${size / 2}" cy="${size / 2}" r="${style.radius}" fill="${style.fillColor}" fill-opacity="${style.fillOpacity}" stroke="${style.color}" stroke-width="${style.weight}"/></svg>`);
      const options = { iconLayout: "default#image", iconImageHref: icon,
        iconImageSize: [size, size], iconImageOffset: [-size / 2, -size / 2], zIndex: 10, zIndexHover: 20 };
      const version = JSON.stringify([position, options]);
      let layer = paths.current.get(key) as ymaps.Placemark | undefined;
      if (!layer) {
        layer = new ymaps.Placemark(position, {}, options);
        instance.geoObjects.add(layer);
        paths.current.set(key, layer);
      } else if (pathVersions.current.get(key) !== version) {
        layer.geometry!.setCoordinates(position); layer.options.set(options);
      }
      pathVersions.current.set(key, version);
      setTooltip(layer, text);
    };
    for (const vehicle of vehicles) {
      const segment = vehicle.segment;
      if (!segment) continue;
      line(`segment:${vehicle.tr_id}`, [[segment.from.lat, segment.from.lon], [segment.to.lat, segment.to.lon]], {
        color: colourOf(vehicle), weight: vehicle.tr_id === selected ? 7 : 4,
        opacity: vehicle.stale ? 0.35 : 0.8,
        dashArray: vehicle.prediction?.status === "ok" ? undefined : "5 5",
      }, `ТС ${vehicle.tr_id}: ${stopLabel(segment.from.address, segment.from.target_stop_id)} → ${stopLabel(segment.to.address, segment.to.target_stop_id)} (схема)`, vehicle.tr_id);
    }
    if (detail) {
      const plan = detail.plan ?? [];
      if (plan.length > 1) line(`plan:${detail.tr_id}`,
        plan.map((visit) => [visit.lat, visit.lon]),
        { color: "#4da3ff", weight: 2, opacity: 0.5, dashArray: "6 6" });
      for (const visit of plan) point(`visit:${detail.tr_id}:${visit.target_stop_id}`, [visit.lat, visit.lon], {
        radius: 4, color: visit.passed ? "#4a5a6d" : "#4da3ff",
        fillColor: visit.passed ? "#2a3542" : "#1d3a58", fillOpacity: 1, weight: 1,
      }, `${sourceTime(visit.planned_at)} · ${stopLabel(visit.address, visit.target_stop_id)}` + (visit.passed ? " · пройдено" : ""));
      const track = (detail.track ?? []).filter((item) => item.lon !== null && item.lat !== null);
      if (track.length > 1) line(`track:${detail.tr_id}`,
        track.map((item) => [item.lat as number, item.lon as number]),
        { color: "#52647d", weight: 2, opacity: 0.8 });
      const prediction = detail.prediction;
      if (prediction?.status === "ok" && prediction.target_lat != null && prediction.target_lon != null) {
        point(`target:${detail.tr_id}:${prediction.target_stop_id}`, [prediction.target_lat, prediction.target_lon], {
          radius: 9, color: "#ffffff", weight: 2,
          fillColor: COLOURS[prediction.risk_level ?? "green"], fillOpacity: 0.85,
        }, `Целевая остановка · план ${sourceTime(prediction.target_planned_at)} · ${signedDelay(prediction.delay_s)}`);
      }
    }
    for (const [key, layer] of paths.current) {
      if (!seen.has(key)) {
        instance.geoObjects.remove(layer); paths.current.delete(key); pathVersions.current.delete(key);
      }
    }
  }, [detail, vehicles, selected, ready]);

  const focus = () => {
    const instance = map.current;
    if (!instance || !selected) return;
    const vehicle = vehicles.find((item) => item.tr_id === selected);
    const points: number[][] = detail?.tr_id === selected
      ? (detail.plan ?? []).map((visit) => [visit.lat, visit.lon]) : [];
    if (vehicle?.segment) points.push([vehicle.segment.from.lat, vehicle.segment.from.lon],
      [vehicle.segment.to.lat, vehicle.segment.to.lon]);
    const marker = markers.current.get(selected);
    if (marker) points.push(marker.getPosition());
    if (points.length) {
      const lat = points.map((point) => point[0]), lon = points.map((point) => point[1]);
      void instance.setBounds([[Math.min(...lat), Math.min(...lon)], [Math.max(...lat), Math.max(...lon)]],
        { zoomMargin: [40, 40, 40, 40], checkZoomRange: true, preciseZoom: false }).then(() => {
          if (map.current === instance && instance.getZoom() > 15) void instance.setZoom(15);
        });
    }
  };

  useEffect(() => {
    if (!ready || !focusRequest || focusRequest.trId !== selected || !vehicles.some((item) => item.tr_id === selected)) return;
    const fullPlan = detail?.tr_id === selected && (detail.plan?.length ?? 0) > 0;
    if (focused.current?.sequence === focusRequest.sequence && (focused.current.fullPlan || !fullPlan)) return;
    focus();
    focused.current = { sequence: focusRequest.sequence, fullPlan };
  }, [focusRequest, selected, detail, vehicles, ready]);

  return (
    <div className="map-wrap">
      <div id="map" ref={container} role="application" aria-label="Карта транспортных средств" />
      <div className="map-buttons">
        <button onClick={focus} disabled={!selected || !ready}>
          Центрировать выбранное
        </button>
      </div>
      {!ready ? <div className="map-note" role="status">
        {mapError ?? "Загрузка Яндекс Карт…"}
        {mapError ? <button onClick={() => setAttempt((value) => value + 1)}>Повторить</button> : null}
      </div> : null}
      <details className="legend">
        <summary>Обозначения карты</summary>
        <div className="legend-content">
        <div className="item">
          <span className="vehicle-colour-key" aria-hidden="true" />
          <span>Обводка ТС — текущее отклонение по GPS; заливка и линия — прогноз к цели.</span>
        </div>
        <div className="legend-note">Серая пунктирная обводка — оценка отсутствует или устарела.
          Белый внешний ореол — выбранное ТС. Текущее отклонение оценено на последней
          распознанной остановке, возраст указан в подсказке. Стрелка ТС показывает курс
          телеметрии; ромб — курс неизвестен.</div>
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
