import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { fitWindow, sourceLabel, timeLabel } from "../replayTime";

interface Props {
  first: number;
  last: number;
  start: number;
  end: number;
  current: number;
  disabled: boolean;
  onRange: (start: number, end: number) => Promise<void>;
  onSeek: (time: number) => Promise<void>;
  onInteractionStart?: () => void;
}

type Drag = { mode: "start" | "end" | "move" | "seek"; origin: number; start: number; end: number; moved: boolean };
const STEPS = [1000, 5000, 15000, 30000, 60000, 300000, 600000, 1800000, 3600000, 7200000, 21600000, 43200000, 86400000];

/** Ruler, editable selection, and an independent source-time playhead. */
export function ReplayTimeline({ first, last, start, end, current, disabled, onRange, onSeek, onInteractionStart }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const [draft, setDraft] = useState<{start: number; end: number} | null>(null);
  const [preview, setPreview] = useState<number | null>(null);
  const [width, setWidth] = useState(1000);
  const [view, setView] = useState({ start: first, end: last });
  const selected = draft ?? { start, end };

  useEffect(() => {
    setView(fitWindow((start + end) / 2, (end - start) / 0.8));
    setDraft(null);
  }, [start, end]);
  useEffect(() => {
    if (!Number.isFinite(current) || drag.current) return;
    setView((previous) => current < previous.start || current > previous.end
      ? fitWindow(current, previous.end - previous.start) : previous);
  }, [current]);
  useEffect(() => {
    if (!track.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(track.current);
    return () => observer.disconnect();
  }, []);

  const span = Math.max(1000, view.end - view.start);
  const percent = (value: number) => (value - view.start) / span * 100;
  const clamp = (value: number, low = -Infinity, high = Infinity) => Math.max(low, Math.min(high, Math.round(value / 1000) * 1000));
  const atPointer = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return clamp(view.start + (event.clientX - rect.left) / rect.width * span);
  };
  const ticks = useMemo(() => {
    const ideal = span / Math.max(2, Math.floor(width / 90));
    const major = STEPS.find((step) => step >= ideal) ?? Math.ceil(ideal / 86400000) * 86400000;
    const minor = Math.max(1000, major / 5);
    const items = [];
    for (let time = Math.ceil(view.start / minor) * minor; time <= view.end; time += minor) {
      items.push({ time, major: time % major === 0, seconds: major < 60000 });
    }
    return items;
  }, [span, width, view.start, view.end]);

  const begin = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0) return;
    onInteractionStart?.();
    const target = (event.target as HTMLElement).closest<HTMLElement>("[data-drag]");
    const mode = (target?.dataset.drag ?? "seek") as Drag["mode"];
    drag.current = { mode, origin: atPointer(event), start, end, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if (mode === "seek") setPreview(atPointer(event));
  };
  const move = (event: PointerEvent<HTMLDivElement>) => {
    const active = drag.current;
    if (!active) return;
    const at = atPointer(event);
    if (Math.abs(at - active.origin) > span / width * 3) active.moved = true;
    if (active.mode === "start") setDraft({ start: clamp(at, -Infinity, active.end - 1000), end: active.end });
    else if (active.mode === "end") setDraft({ start: active.start, end: clamp(at, active.start + 1000) });
    else if (active.mode === "move") {
      const delta = at - active.origin;
      setDraft({ start: active.start + delta, end: active.end + delta });
    } else setPreview(at);
  };
  const finish = (event: PointerEvent<HTMLDivElement>) => {
    const active = drag.current;
    if (!active) return;
    drag.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (active.mode === "seek" || (active.mode === "move" && !active.moved)) {
      void onSeek(atPointer(event));
    } else if (draft) void onRange(draft.start, draft.end);
    setDraft(null);
    setPreview(null);
  };
  const keyboardRange = (edge: "start" | "end", key: string) => {
    const delta = key === "ArrowLeft" ? -60000 : key === "ArrowRight" ? 60000 : 0;
    if (!delta || disabled) return;
    if (edge === "start") void onRange(clamp(start + delta, -Infinity, end - 1000), end);
    else void onRange(start, clamp(end + delta, start + 1000));
  };
  const playhead = preview ?? current;
  const visible = playhead >= view.start && playhead <= view.end;

  return (
    <div className="ruler-row">
      <div ref={track} className={`time-ruler${disabled ? " is-disabled" : ""}`}
        role="group" aria-label="Линейка исторического времени"
        onPointerDown={begin} onPointerMove={move} onPointerUp={finish}
        onPointerCancel={() => { drag.current = null; setDraft(null); setPreview(null); }}
        onKeyDown={(event) => {
          if ((event.target as HTMLElement).dataset.keyboard !== "seek" || disabled) return;
          const value = event.key === "Home" ? start : event.key === "End" ? end
            : event.key === "ArrowLeft" ? clamp(current - 1000)
            : event.key === "ArrowRight" ? clamp(current + 1000) : null;
          if (value !== null) { event.preventDefault(); void onSeek(value); }
        }}>
        <div className="ruler-seek-control" data-keyboard="seek" role="slider"
          tabIndex={disabled ? -1 : 0} aria-label="Историческая временная шкала"
          aria-valuemin={Math.min(view.start, current)} aria-valuemax={Math.max(view.end, current)} aria-valuenow={current}
          aria-valuetext={sourceLabel(current)} aria-disabled={disabled} />
        <div className="ruler-selection" data-drag="move" title="Перетащите диапазон; нажмите для перемотки"
          style={{ display: selected.end < view.start || selected.start > view.end ? "none" : undefined,
            left: `${Math.max(0, percent(selected.start))}%`, right: `${Math.max(0, 100 - percent(selected.end))}%` }} />
        {view.start < first ? <span className="ruler-no-data" title="Нет телеметрии"
          style={{ left: 0, width: `${Math.min(100, percent(first))}%` }} /> : null}
        {view.end > last ? <span className="ruler-no-data" title="Нет телеметрии"
          style={{ right: 0, width: `${Math.min(100, 100 - percent(last))}%` }} /> : null}
        {ticks.map((tick) => (
          <span key={tick.time} className={`ruler-tick${tick.major ? " major" : ""}`}
            style={{ left: `${percent(tick.time)}%` }}>
            {tick.major ? <span>{span >= 86400000 ? sourceLabel(tick.time).slice(5, 16) : tick.seconds ? timeLabel(tick.time) : timeLabel(tick.time).slice(0, 5)}</span> : null}
          </span>
        ))}
        {(["start", "end"] as const).map((edge) => {
          const value = selected[edge];
          return value >= view.start && value <= view.end ? (
            <div key={edge} className="ruler-handle" data-drag={edge} role="slider"
              tabIndex={disabled ? -1 : 0} aria-label={edge === "start" ? "Начало выделенного диапазона" : "Конец выделенного диапазона"}
              aria-valuemin={edge === "start" ? Math.min(view.start, value) : start + 1000} aria-valuemax={edge === "start" ? end - 1000 : Math.max(view.end, value)}
              aria-valuenow={value} aria-valuetext={sourceLabel(value)} aria-disabled={disabled}
              title={sourceLabel(value)} style={{ left: `clamp(3px, ${percent(value)}%, calc(100% - 3px))` }}
              onKeyDown={(event) => { event.stopPropagation(); if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); keyboardRange(edge, event.key); } }} />
          ) : null;
        })}
        {visible ? <span className="ruler-playhead" title={`Текущее время: ${sourceLabel(playhead)}`}
          style={{ left: `${percent(playhead)}%` }} /> : null}
      </div>
      <div className="ruler-zoom" role="group" aria-label="Масштаб временной шкалы"
        onPointerDown={onInteractionStart} onKeyDown={onInteractionStart}>
        <button disabled={disabled} aria-label="Уменьшить масштаб времени" onClick={() => setView(fitWindow((view.start + view.end) / 2, span * 2))}>−</button>
        <button disabled={disabled || span <= 60000} aria-label="Увеличить масштаб времени" onClick={() => setView(fitWindow((view.start + view.end) / 2, Math.max(60000, span / 2)))}>+</button>
        <button disabled={disabled} aria-label="Показать более раннее время" onClick={() => setView(fitWindow((view.start + view.end) / 2 - span / 2, span))}>‹</button>
        <button disabled={disabled} aria-label="Показать более позднее время" onClick={() => setView(fitWindow((view.start + view.end) / 2 + span / 2, span))}>›</button>
        <button disabled={disabled} title="Показать текущий момент на шкале" onClick={() => setView(fitWindow(clamp(current), span))}>К текущему</button>
      </div>
    </div>
  );
}
