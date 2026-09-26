import { useEffect, useState } from "react";
import type { ReplayAction } from "../api";
import type { Status } from "../types";

interface Props {
  status: Status | null;
  onCommand: (action: ReplayAction, speed?: number, startAt?: string) => Promise<void>;
  pending: boolean;
  showSystem: boolean;
  onToggleSystem: () => void;
  error: string | null;
}

const SPEEDS = [1, 10, 30, 120, 300];

// UTC is used only as a numeric encoding of naive calendar fields. No local timezone
// conversion is applied, and the timestamp sent to Backend has no offset or Z suffix.
function sourceMs(value: string | null | undefined): number {
  return value ? Date.parse(value.replace(" ", "T").slice(0, 23) + "Z") : NaN;
}

function sourceInput(value: number): string {
  return Number.isFinite(value) ? new Date(value).toISOString().slice(0, 19) : "";
}

function sourceLabel(value: number): string {
  return sourceInput(value).replace("T", " ");
}

interface NavigationRange {
  sourceFirst: number;
  sourceLast: number;
  start: number;
  end: number;
}

/** Historical navigation rebuilds the causal prefix and leaves the clock paused. */
export function ReplayControls({ status, onCommand, pending, showSystem, onToggleSystem, error }: Props) {
  const replay = status?.mode === "replay" && status?.replay?.control_enabled;
  const clock = status?.clock;
  const first = Math.ceil(sourceMs(clock?.source_window?.first) / 1000) * 1000;
  const last = Math.floor(sourceMs(clock?.source_window?.last) / 1000) * 1000;
  const current = sourceMs(clock?.source_time);
  const hasPeriod = Number.isFinite(first) && Number.isFinite(last) && first <= last;
  const [selection, setSelection] = useState<NavigationRange | null>(null);
  const range = selection?.sourceFirst === first && selection?.sourceLast === last
    ? selection : { start: first, end: last };
  const [draft, setDraft] = useState<number | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const position = hasPeriod ? Math.max(range.start, Math.min(range.end, draft ?? current)) : 0;

  useEffect(() => {
    setDraft(null);
    setLocalError(null);
  }, [status?.run_id]);

  const seek = async (value: number) => {
    if (!Number.isFinite(value) || !hasPeriod || value < range.start || value > range.end) {
      setLocalError("Выберите время внутри выбранного диапазона");
      return;
    }
    setLocalError(null);
    await onCommand("seek", undefined, sourceInput(value).replace("T", " "));
    setDraft(null);
  };

  const applyRange = async (start: number, end: number) => {
    if (!hasPeriod || !Number.isFinite(start) || !Number.isFinite(end)
      || start < first || end > last || start >= end) {
      setLocalError("Начало должно быть раньше конца, обе границы — внутри доступного периода");
      return;
    }
    setSelection({ sourceFirst: first, sourceLast: last, start, end });
    setDraft(null);
    setLocalError(null);
    // A new navigation interval should display the real position, not a clamped fiction.
    if (current < start || current > end) {
      await onCommand("seek", undefined, sourceLabel(start));
    }
  };

  return (
    <footer className="footer">
      <span className="footer-mode">
        {status?.mode === "ndtp"
          ? "Живой приём NDTP: виртуальные часы следуют за потоком"
          : "Исторический replay по времени источника"}
      </span>
      {replay ? (
        <div className="replay-actions">
          <button disabled={pending} onClick={() => onCommand(clock?.paused ? "start" : "pause")}>
            {clock?.paused ? "Продолжить" : "Пауза"}
          </button>
          <div className="speed-buttons" role="group" aria-label="Скорость воспроизведения">
            {SPEEDS.map((speed) => (
              <button
                key={speed}
                disabled={pending}
                className={clock?.speed === speed ? "active" : ""}
                onClick={() => onCommand("speed", speed)}
              >
                {speed}×
              </button>
            ))}
          </div>
          <button disabled={pending} onClick={() => onCommand("reset")}>Сброс прогона</button>
        </div>
      ) : (
        <span className="hint">управление replay недоступно в этом режиме</span>
      )}
      <div className="footer-status">
        {error || localError ? <span role="alert" className="risk-red">{localError ?? error}</span> : null}
        <button onClick={onToggleSystem} aria-pressed={showSystem}>
          {showSystem ? "Скрыть панель системы" : "Панель системы"}
        </button>
      </div>
      {replay && hasPeriod ? (
        <div className="replay-navigation" aria-label="Перемотка исторического времени">
          <form className="replay-range" key={`${first}:${last}:${range.start}:${range.end}`} onSubmit={(event) => {
            event.preventDefault();
            const fields = new FormData(event.currentTarget);
            void applyRange(sourceMs(String(fields.get("from"))), sourceMs(String(fields.get("to"))));
          }}>
            <label>Начало диапазона
              <input name="from" type="datetime-local" step="1" required
                min={sourceInput(first)} max={sourceInput(last)} defaultValue={sourceInput(range.start)} disabled={pending} />
            </label>
            <label>Конец диапазона
              <input name="to" type="datetime-local" step="1" required
                min={sourceInput(first)} max={sourceInput(last)} defaultValue={sourceInput(range.end)} disabled={pending} />
            </label>
            <button disabled={pending} type="submit">Применить диапазон</button>
            <button disabled={pending} type="button" onClick={() => {
              setSelection(null); setDraft(null); setLocalError(null);
            }}>Весь период</button>
            <span className="hint range-available">Данные: {sourceLabel(first)} — {sourceLabel(last)}</span>
          </form>
          <div className="replay-timeline">
            <div className="timeline-labels">
              <span>{sourceLabel(range.start)}</span>
              <strong>{pending ? "Применяем команду…" : `Время: ${sourceLabel(draft ?? current)}`}</strong>
              <span>{sourceLabel(range.end)}</span>
            </div>
            <input
              type="range"
              aria-label="Историческая временная шкала"
              min={range.start}
              max={range.end}
              step={1000}
              value={Number.isFinite(position) ? position : first}
              disabled={pending}
              onChange={(event) => { setDraft(Number(event.target.value)); setLocalError(null); }}
              onPointerUp={(event) => { if (!pending) void seek(Number(event.currentTarget.value)); }}
              onPointerCancel={() => setDraft(null)}
              onKeyUp={(event) => {
                if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
                  void seek(Number(event.currentTarget.value));
                }
              }}
            />
          </div>
          <form className="replay-exact" onSubmit={(event) => {
            event.preventDefault();
            const value = new FormData(event.currentTarget).get("start_at");
            void seek(sourceMs(typeof value === "string" ? value : ""));
          }}>
            <label htmlFor="replay-time">Перейти к дате и времени</label>
            <div className="replay-exact-fields">
              <input id="replay-time" name="start_at" key={`${status?.run_id}:${range.start}:${range.end}`} type="datetime-local" step="1" required
                min={sourceInput(range.start)} max={sourceInput(range.end)} defaultValue={sourceInput(current)} disabled={pending}
                onChange={() => setLocalError(null)} />
              <button disabled={pending} type="submit">Перейти</button>
            </div>
          </form>
          {current < range.start || current > range.end ? (
            <span className="hint replay-seek-note">Текущее время вне выбранного диапазона.
              <button disabled={pending} onClick={() => void seek(range.start)}>К началу диапазона</button>
            </span>
          ) : null}
          <span className="hint replay-seek-note">Перемотка очищает прошлый прогон и ставит время на паузу. Время — как в датасете.</span>
        </div>
      ) : null}
    </footer>
  );
}
