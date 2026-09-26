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

export function fitWindow(center: number, span: number, first: number, last: number) {
  const size = Math.min(span, last - first);
  const start = Math.max(first, Math.min(last - size, center - size / 2));
  return { start: Math.round(start / 1000) * 1000, end: Math.round((start + size) / 1000) * 1000 };
}
