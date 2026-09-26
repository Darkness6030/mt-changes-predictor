import L from "leaflet";
import { signedDelay, sourceTime } from "./format";
import { deviationColour, deviationTooltip, forecastColour } from "./mapPresentation";
import { headingDegrees, vehicleOffsets } from "./vehicleHeading";
import type { Vehicle } from "./types";

/** Persistent canvas arrow: course geometry, current outline, forecast fill, selection halo. */
export class VehicleMarker {
  private readonly group: L.LayerGroup;
  private readonly halo: L.Polygon;
  private readonly ring: L.Polygon;
  private readonly core: L.Polygon;
  private tooltipText = "";
  private appearance = "";
  private position: L.LatLng;
  private heading: number | null;
  private readonly onZoom = () => this.updateGeometry();

  constructor(vehicle: Vehicle, private readonly map: L.Map, onSelect: (trId: string) => void) {
    this.position = L.latLng(vehicle.position!.lat!, vehicle.position!.lon!);
    this.heading = headingDegrees(vehicle.position!.heading_deg);
    this.halo = L.polygon([], {
      weight: 11, color: "#ffffff", opacity: 0, fill: false, interactive: false,
      lineJoin: "round",
    });
    this.ring = L.polygon([], {
      weight: 7, fill: false, interactive: false, lineJoin: "round",
    });
    this.core = L.polygon([], { weight: 1.5, color: "#17202b", lineJoin: "round" })
      .on("click", () => onSelect(vehicle.tr_id));
    this.core.on("tooltipopen", () => {
      const tooltip = this.core.getTooltip();
      const element = tooltip?.getElement();
      if (element) {
        element.style.maxWidth = `${Math.max(100, Math.min(290, map.getSize().x / 2 - 24))}px`;
        tooltip!.update();
      }
    });
    this.group = L.layerGroup([this.halo, this.ring, this.core]).addTo(map);
    this.updateGeometry();
    map.on("zoomend", this.onZoom);
    this.update(vehicle, false);
  }

  update(vehicle: Vehicle, selected: boolean): void {
    const position = L.latLng(vehicle.position!.lat!, vehicle.position!.lon!);
    const heading = headingDegrees(vehicle.position!.heading_deg);
    if (!this.position.equals(position) || this.heading !== heading) {
      this.position = position;
      this.heading = heading;
      this.updateGeometry();
    }
    const current = vehicle.current_deviation;
    const outdated = vehicle.stale || vehicle.prediction?.status === "ml_unavailable";
    const appearance = `${selected}:${deviationColour(current)}:${current?.status}:${forecastColour(vehicle)}:${outdated}`;
    if (appearance !== this.appearance) {
      this.halo.setStyle({ opacity: selected ? 0.95 : 0 });
      this.ring.setStyle({
        color: deviationColour(current), opacity: 1,
        dashArray: current?.status === "ok" ? undefined : "3 3",
      });
      this.core.setStyle({ fillColor: forecastColour(vehicle), fillOpacity: outdated ? 0.35 : 0.9 });
      this.appearance = appearance;
    }
    const prediction = vehicle.prediction;
    const forecast = prediction && ["ok", "ml_unavailable"].includes(prediction.status)
      ? `Прогноз: ${signedDelay(prediction.delay_s)} · цель ${sourceTime(prediction.target_planned_at)}` +
        (prediction.status === "ml_unavailable" ? " · прошлый, ML недоступен" : "")
      : "Прогноз: нет данных";
    const course = this.heading === null ? "Курс неизвестен" :
      `Курс: ${Math.round(this.heading)}°` + (vehicle.position?.speed_kmh === 0 ? " · ТС стоит" : "");
    const text = `ТС ${vehicle.tr_id}\n${course}\n${deviationTooltip(current)}\n${forecast}` +
      (vehicle.stale ? "\nДанные позиции устарели" : "");
    if (text !== this.tooltipText) {
      const label = document.createElement("span");
      label.textContent = text;
      if (this.core.getTooltip()) this.core.setTooltipContent(label);
      else this.core.bindTooltip(label, { direction: "auto", className: "vehicle-tooltip" });
      this.tooltipText = text;
    }
  }

  private updateGeometry(): void {
    const center = this.map.latLngToLayerPoint(this.position);
    const geometry = vehicleOffsets(this.heading).map(([x, y]) =>
      this.map.layerPointToLatLng(L.point(center.x + x, center.y + y)));
    this.halo.setLatLngs(geometry);
    this.ring.setLatLngs(geometry);
    this.core.setLatLngs(geometry);
  }

  bringToFront(): void {
    this.halo.bringToFront();
    this.ring.bringToFront();
    this.core.bringToFront();
  }

  getLatLng(): L.LatLng { return this.position; }
  remove(): void {
    this.map.off("zoomend", this.onZoom);
    this.group.remove();
  }
}
