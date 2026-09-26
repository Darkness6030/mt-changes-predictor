/** Telemetry course: north = 0°, clockwise. Null must never become a northbound arrow. */
export function headingDegrees(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value >= 0 && value <= 360
    ? value % 360 : null;
}

/** Screen-space silhouette, in pixels; the first vertex is the arrow's tip. */
export function vehicleOffsets(heading: number | null): [number, number][] {
  if (heading === null) return [[0, -10], [10, 0], [0, 10], [-10, 0]];
  const angle = heading * Math.PI / 180;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  return [[0, -13], [10, 11], [0, 6], [-10, 11]].map(([x, y]) =>
    [x * cos - y * sin, x * sin + y * cos]);
}
