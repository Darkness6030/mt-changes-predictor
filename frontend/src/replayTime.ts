// Calendar fields are encoded numerically without converting the dataset's timezone.
export function sourceMs(value: string | null | undefined): number {
  return value ? Date.parse(value.replace(" ", "T").slice(0, 23) + "Z") : NaN;
}

export function sourceInput(value: number): string {
  return Number.isFinite(value) ? new Date(value).toISOString().slice(0, 19) : "";
}

export function sourceLabel(value: number): string {
  return sourceInput(value).replace("T", " ");
}

export function timeLabel(value: number): string {
  return sourceInput(value).slice(11);
}

export function fitWindow(center: number, span: number) {
  const start = Math.round((center - span / 2) / 1000) * 1000;
  return { start, end: start + Math.round(span / 1000) * 1000 };
}
