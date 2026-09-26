import { useEffect, useState } from "react";
import type { ReplayAction } from "../api";
import { sourceTime } from "../format";
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

/** Historical navigation rebuilds the causal prefix and leaves the clock paused. */
export function ReplayControls({ status, onCommand, pending, showSystem, onToggleSystem, error }: Props) {
  const replay = status?.mode === "replay" && status?.replay?.control_enabled;
  const clock = status?.clock;
  const first = Math.ceil(sourceMs(clock?.source_window?.first) / 1000) * 1000;
  const last = Math.floor(sourceMs(clock?.source_window?.last) / 1000) * 1000;
  const current = sourceMs(clock?.source_time);
  const hasPeriod = Number.isFinite(first) && Number.isFinite(last) && first <= last;
  const [draft, setDraft] = useState<number | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const position = hasPeriod ? Math.max(first, Math.min(last, draft ?? current)) : 0;

  useEffect(() => {
    setDraft(null);
    setLocalError(null);
  }, [status?.run_id]);

  const seek = async (value: number) => {
    if (!Number.isFinite(value) || !hasPeriod || value < first || value > last) {
      setLocalError("Выберите время внутри периода исторических данных");
      return;
    }
    setLocalError(null);
    await onCommand("seek", undefined, sourceInput(value).replace("T", " "));
    setDraft(null);
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
        <span className="source-period">
          период {sourceTime(clock?.source_window?.first)} — {sourceTime(clock?.source_window?.last)}
        </span>
        <button onClick={onToggleSystem} aria-pressed={showSystem}>
          {showSystem ? "Скрыть панель системы" : "Панель системы"}
        </button>
      </div>
      {replay && hasPeriod ? (
        <div className="replay-navigation" aria-label="Перемотка исторического времени">
          <div className="replay-timeline">
            <div className="timeline-labels">
              <span>{sourceTime(clock?.source_window?.first)}</span>
              <strong>{pending ? "Применяем команду…" : `Время: ${sourceInput(position).replace("T", " ")}`}</strong>
              <span>{sourceTime(clock?.source_window?.last)}</span>
            </div>
            <input
              type="range"
              aria-label="Историческая временная шкала"
              min={first}
              max={last}
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
              <input id="replay-time" name="start_at" key={status?.run_id} type="datetime-local" step="1" required
                min={sourceInput(first)} max={sourceInput(last)} defaultValue={sourceInput(current)} disabled={pending}
                onChange={() => setLocalError(null)} />
              <button disabled={pending} type="submit">Перейти</button>
            </div>
          </form>
          <span className="hint replay-seek-note">Перемотка очищает прошлый прогон и ставит время на паузу. Время — как в датасете.</span>
        </div>
      ) : null}
    </footer>
  );
}
