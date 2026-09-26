import { signedDelay, sourceTime } from "./format";
import { deviationColour, deviationTooltip, forecastColour } from "./mapPresentation";
import { headingDegrees, vehicleOffsets } from "./vehicleHeading";
import { hintHtml, svgImage } from "./yandexMaps";
import type { Vehicle } from "./types";

/** Stable Yandex placemark with a screen-sized SVG arrow and native hover hint. */
export class VehicleMarker {
  private readonly placemark: ymaps.Placemark;
  private tooltipText = "";
  private appearance = "";
  private heading: number | null = null;

  constructor(vehicle: Vehicle, private readonly map: ymaps.Map, onSelect: (trId: string) => void) {
    this.placemark = new ymaps.Placemark([vehicle.position!.lat!, vehicle.position!.lon!], {}, {
      iconLayout: "default#image", iconImageSize: [44, 44], iconImageOffset: [-22, -22],
      zIndex: 1000, zIndexHover: 2000, cursor: "pointer",
      hideIconOnBalloonOpen: false,
    });
    this.placemark.events.add("click", () => onSelect(vehicle.tr_id));
    map.geoObjects.add(this.placemark);
    this.update(vehicle, false);
  }

  update(vehicle: Vehicle, selected: boolean): void {
    const position = [vehicle.position!.lat!, vehicle.position!.lon!];
    const previous = this.getPosition();
    if (position[0] !== previous[0] || position[1] !== previous[1]) {
      this.placemark.geometry!.setCoordinates(position);
    }
    this.heading = headingDegrees(vehicle.position!.heading_deg);
    const current = vehicle.current_deviation;
    const outdated = vehicle.stale || vehicle.prediction?.status === "ml_unavailable";
    const appearance = `${this.heading}:${selected}:${deviationColour(current)}:${current?.status}:${forecastColour(vehicle)}:${outdated}`;
    if (appearance !== this.appearance) {
      const points = vehicleOffsets(this.heading).map(([x, y]) => `${x + 22},${y + 22}`).join(" ");
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44">
        <g stroke-linejoin="round">
          <polygon points="${points}" fill="none" stroke="white" stroke-width="11" opacity="${selected ? 0.95 : 0}"/>
          <polygon points="${points}" fill="none" stroke="${deviationColour(current)}" stroke-width="7" stroke-dasharray="${current?.status === "ok" ? "none" : "3 3"}"/>
          <polygon points="${points}" stroke="#17202b" stroke-width="1.5" fill="${forecastColour(vehicle)}" fill-opacity="${outdated ? 0.35 : 0.9}"/>
        </g></svg>`;
      this.placemark.options.set({ iconImageHref: svgImage(svg), zIndex: selected ? 1500 : 1000 });
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
      this.placemark.properties.set("hintContent", hintHtml(text));
      this.tooltipText = text;
    }
  }

  getPosition(): number[] { return this.placemark.geometry!.getCoordinates()!; }
  remove(): void { this.map.geoObjects.remove(this.placemark); }
}
