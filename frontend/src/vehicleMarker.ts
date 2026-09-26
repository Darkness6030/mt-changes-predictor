import L from "leaflet";
import { signedDelay, sourceTime } from "./format";
import { deviationColour, deviationTooltip, forecastColour } from "./mapPresentation";
import type { Vehicle } from "./types";

/** Three persistent canvas circles: selection halo, current deviation ring, forecast core. */
export class VehicleMarker {
  private readonly group: L.LayerGroup;
  private readonly halo: L.CircleMarker;
  private readonly ring: L.CircleMarker;
  private readonly core: L.CircleMarker;
  private tooltipText = "";
  private appearance = "";

  constructor(vehicle: Vehicle, map: L.Map, onSelect: (trId: string) => void) {
    const position: L.LatLngTuple = [vehicle.position!.lat!, vehicle.position!.lon!];
    this.halo = L.circleMarker(position, {
      radius: 16, weight: 2, color: "#ffffff", opacity: 0, fill: false, interactive: false,
    });
    this.ring = L.circleMarker(position, {
      radius: 12, weight: 3, fill: false, interactive: false,
    });
    this.core = L.circleMarker(position, { radius: 8, weight: 1.5, color: "#17202b" })
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
    this.update(vehicle, false);
  }

  update(vehicle: Vehicle, selected: boolean): void {
    const position = L.latLng(vehicle.position!.lat!, vehicle.position!.lon!);
    if (!this.core.getLatLng().equals(position)) {
      this.halo.setLatLng(position);
      this.ring.setLatLng(position);
      this.core.setLatLng(position);
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
    const text = `ТС ${vehicle.tr_id}\n${deviationTooltip(current)}\n${forecast}` +
      (vehicle.stale ? "\nДанные позиции устарели" : "");
    if (text !== this.tooltipText) {
      const label = document.createElement("span");
      label.textContent = text;
      if (this.core.getTooltip()) this.core.setTooltipContent(label);
      else this.core.bindTooltip(label, { direction: "auto", className: "vehicle-tooltip" });
      this.tooltipText = text;
    }
  }

  bringToFront(): void {
    this.halo.bringToFront();
    this.ring.bringToFront();
    this.core.bringToFront();
  }

  getLatLng(): L.LatLng { return this.core.getLatLng(); }
  remove(): void { this.group.remove(); }
}
