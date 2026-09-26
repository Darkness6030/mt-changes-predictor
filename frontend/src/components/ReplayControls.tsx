import { useEffect, useRef, useState, type ReactNode } from "react";
import type { ReplayAction } from "../api";
import { fitWindow, sourceInput, sourceLabel, sourceMs, timeLabel } from "../replayTime";
import type { Status } from "../types";
import type { useHistoryView } from "../useHistoryView";
import { ReplayTimeline } from "./ReplayTimeline";

interface Props {
  status: Status | null;
  onCommand: (action: ReplayAction, speed?: number, startAt?: string) => Promise<void>;
  pending: boolean;
  showSystem: boolean;
  onToggleSystem: () => void;
  sourceControl?: ReactNode;
  historyView: ReturnType<typeof useHistoryView>;
}
interface NavigationRange { key: string; start: number; end: number }
const SPEEDS = [1, 10, 30, 60, 120, 300];
const PRESETS = [{ label: "15м", span: 900000 }, { label: "1ч", span: 3600000 }, { label: "6ч", span: 21600000 }];

/** Compact time toolbar and ruler. All timestamps retain dataset calendar fields. */
export function ReplayControls({ status, onCommand, pending, showSystem, onToggleSystem, sourceControl, historyView }: Props) {
  const replay = status?.mode === "replay" && status.replay?.control_enabled;
  const live = status?.mode === "ndtp";
  const clock = status?.clock;
  const first = sourceMs(live ? status?.history?.first : clock?.source_window?.first);
  const last = sourceMs(live ? status?.history?.last : clock?.source_window?.last);
  const current = sourceMs(live && historyView.active ? historyView.at : clock?.source_time);
  const hasPeriod = Number.isFinite(first) && Number.isFinite(last) && first <= last;
  const rangeKey = live ? status.run_id : `${first}:${last}`;
  const [selection, setSelection] = useState<NavigationRange | null>(null);
  const range = selection?.key === rangeKey ? selection : {
    start: Math.floor(first / 1000) * 1000,
    end: Math.max(Math.ceil(last / 1000) * 1000, Math.floor(first / 1000) * 1000 + 60000),
  };
  const [editor, setEditor] = useState<"range" | "time" | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const editorRoot = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!editor) return;
    const outside = (event: PointerEvent) => {
      if (!editorRoot.current?.contains(event.target as Node)) setEditor(null);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [editor]);

  const seek = async (value: number) => {
    if (!hasPeriod || !Number.isFinite(value)) {
      setLocalError("Укажите корректную дату и время"); return;
    }
    setLocalError(null);
    setEditor(null);
    if (live) {
      setSelection({ key: rangeKey, ...range });
      historyView.select(sourceLabel(value));
    } else await onCommand("seek", undefined, sourceLabel(value));
  };
  const applyRange = async (start: number, end: number) => {
    if (!hasPeriod || !Number.isFinite(start) || !Number.isFinite(end) || start >= end) {
      setLocalError("Начало диапазона должно быть раньше конца"); return;
    }
    setSelection({ key: rangeKey, start, end });
    setLocalError(null);
    setEditor(null);
    if (current < start || current > end) {
      if (live) historyView.select(sourceLabel(start));
      else await onCommand("seek", undefined, sourceLabel(start));
    }
  };
  const shiftRange = (direction: number) => {
    const size = range.end - range.start;
    const shifted = fitWindow((range.start + range.end) / 2 + direction * size, size);
    void applyRange(shifted.start, shifted.end);
  };

  return (
    <footer className="footer replay-footer">
      <div className="replay-toolbar">
        {(replay || live) && hasPeriod ? (
          <>
            <div className="time-editor-root" ref={editorRoot} onKeyDown={(event) => { if (event.key === "Escape") setEditor(null); }}>
              <button className="time-range-button" aria-label="Выбрать диапазон даты и времени" aria-expanded={editor === "range"}
                disabled={pending} onClick={() => {
                  if (live) setSelection({ key: rangeKey, ...range });
                  setLocalError(null); setEditor(editor === "range" ? null : "range");
                }}>
                <span>{sourceLabel(range.start)} — {sourceLabel(range.end)}</span><span className="chevron">⌄</span>
              </button>
              <button className="current-time-button" aria-label="Перейти к точному времени" aria-expanded={editor === "time"}
                disabled={pending} onClick={() => {
                  if (live) setSelection({ key: rangeKey, ...range });
                  setLocalError(null); setEditor(editor === "time" ? null : "time");
                }}>
                <span className="playhead-dot" />{timeLabel(current)}
              </button>
              {editor ? (
                <div className="time-editor" role="dialog" aria-label={editor === "range" ? "Выбор диапазона" : "Точное время"}>
                  <div className="time-editor-heading"><strong>{editor === "range" ? "Диапазон навигации" : "Перейти к моменту"}</strong>
                    <button aria-label="Закрыть выбор времени" onClick={() => setEditor(null)}>×</button></div>
                  {editor === "range" ? (
                    <form key={`${range.start}:${range.end}`} onSubmit={(event) => {
                      event.preventDefault(); const values = new FormData(event.currentTarget);
                      void applyRange(sourceMs(String(values.get("from"))), sourceMs(String(values.get("to"))));
                    }}>
                      <label>Начало диапазона<input name="from" type="datetime-local" step="1" required
                        defaultValue={sourceInput(range.start)} /></label>
                      <label>Конец диапазона<input name="to" type="datetime-local" step="1" required
                        defaultValue={sourceInput(range.end)} /></label>
                      <button type="submit">Применить диапазон</button>
                    </form>
                  ) : (
                    <form key={`${status?.run_id}:${range.start}:${range.end}`} onSubmit={(event) => {
                      event.preventDefault(); const values = new FormData(event.currentTarget);
                      void seek(sourceMs(String(values.get("start_at"))));
                    }}>
                      <label>Перейти к дате и времени<input name="start_at" type="datetime-local" step="1" required
                        defaultValue={sourceInput(current)} /></label>
                      <button type="submit">Перейти</button>
                    </form>
                  )}
                  {localError ? <p className="risk-red" role="alert">{localError}</p> : null}
                  <p className="hint">Данные: {sourceLabel(first)} — {sourceLabel(last)}</p>
                  {live ? <p className="hint">Записанные снимки текущего прогона. Храним до {Math.round((status?.history?.window_s ?? 7200) / 60)} мин. времени источника в пределах памяти. Часовой пояс источника не преобразуется.</p> : null}
                </div>
              ) : null}
            </div>
            <div className="range-shortcuts" role="group" aria-label="Навигация по диапазонам">
              <button disabled={pending} aria-label="Предыдущий диапазон" onClick={() => shiftRange(-1)}>‹</button>
              <button disabled={pending} aria-label="Следующий диапазон" onClick={() => shiftRange(1)}>›</button>
              {PRESETS.map((preset) => <button key={preset.label} disabled={pending}
                className={range.end - range.start === preset.span ? "active" : ""}
                onClick={() => { const next = fitWindow(current, preset.span); void applyRange(next.start, next.end); }}>{preset.label}</button>)}
              <button disabled={pending} className={!selection || selection.key !== rangeKey ? "active" : ""}
                onClick={() => live ? setSelection(null) : void applyRange(first, last)}>Всё</button>
            </div>
          </>
        ) : <span className="hint">{live ? "NDTP · ждём первые данные для истории…" : "Загрузка исторического периода…"}</span>}
        <div className="replay-playback">
          {live ? <>
            <span className="history-view-note" role="status" title={historyView.frame
              ? `Снимок ${historyView.frame.recorded_at}; отставание от выбранного момента ${historyView.frame.lag_s.toFixed(1)} с. Без интерполяции и пересчёта прогнозов.`
              : historyView.active ? historyView.error ?? "Загрузка сохранённого снимка"
                : "Перемотка не останавливает приём и прогнозы; система показывает текущий поток"}>
              {historyView.active ? historyView.loading ? "Загрузка снимка…" : historyView.frame
                ? `Снимок ${timeLabel(sourceMs(historyView.frame.recorded_at))}` : "История · нет снимка" : "● В потоке"}
            </span>
            {historyView.active ? <button className="return-live" disabled={pending} onClick={() => {
              historyView.follow(); setSelection(null);
            }}>К потоку →</button> : <button disabled={pending || !status?.history?.last}
              title="Зафиксировать снимок для просмотра; приём потока продолжается" onClick={() => {
              setSelection({ key: rangeKey, ...range });
              historyView.select(status!.history!.last!);
            }}>Ⅱ Просмотр</button>}
          </> : null}
          {replay ? <>
            <button disabled={pending} onClick={() => onCommand(clock?.paused ? "start" : "pause")}>
              {clock?.paused ? "▷ Продолжить" : "Ⅱ Пауза"}</button>
            <select aria-label="Скорость воспроизведения" value={clock?.speed ?? 60} disabled={pending}
              onChange={(event) => onCommand("speed", Number(event.target.value))}>
              {!SPEEDS.includes(clock?.speed ?? 60) ? <option value={clock?.speed}>{clock?.speed}×</option> : null}
              {SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}×</option>)}
            </select>
            <button disabled={pending} aria-label="Сброс прогона" title="Сброс прогона" onClick={() => onCommand("reset")}>↺</button>
          </> : null}
          {sourceControl}
          <button onClick={onToggleSystem} aria-pressed={showSystem}>{showSystem ? "Скрыть систему" : "Система"}</button>
        </div>
      </div>
      {(replay || live) && hasPeriod ? <ReplayTimeline key={live ? status.run_id : "csv"} first={first} last={last} start={range.start} end={range.end}
        onInteractionStart={live ? () => setSelection({ key: rangeKey, ...range }) : undefined}
        current={current} disabled={pending} onRange={applyRange} onSeek={seek} /> : null}
    </footer>
  );
}
