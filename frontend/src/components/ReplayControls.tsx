import { sourceTime } from "../format";
import type { Status } from "../types";

interface Props {
  status: Status | null;
  onCommand: (action: "start" | "pause" | "reset" | "speed", speed?: number) => void;
  showSystem: boolean;
  onToggleSystem: () => void;
  error: string | null;
}

const SPEEDS = [1, 10, 30, 120, 300];

/** Demo-only replay control. In live NDTP mode the buttons are disabled, not hidden. */
export function ReplayControls({ status, onCommand, showSystem, onToggleSystem, error }: Props) {
  const replay = status?.mode === "replay" && status?.replay?.control_enabled;
  const clock = status?.clock;
  return (
    <footer className="footer">
      <span className="footer-mode">
        {status?.mode === "ndtp"
          ? "Живой приём NDTP: виртуальные часы следуют за потоком"
          : "Исторический replay по времени источника"}
      </span>
      {replay ? (
        <div className="replay-actions">
          <button onClick={() => onCommand(clock?.paused ? "start" : "pause")}>
            {clock?.paused ? "Продолжить" : "Пауза"}
          </button>
          <div className="speed-buttons" role="group" aria-label="Скорость воспроизведения">
            {SPEEDS.map((speed) => (
              <button
                key={speed}
                className={clock?.speed === speed ? "active" : ""}
                onClick={() => onCommand("speed", speed)}
              >
                {speed}×
              </button>
            ))}
          </div>
          <button onClick={() => onCommand("reset")}>Сброс прогона</button>
        </div>
      ) : (
        <span className="hint">управление replay недоступно в этом режиме</span>
      )}
      <div className="footer-status">
        {error ? <span className="risk-red">{error}</span> : null}
        <span className="source-period">
          период {sourceTime(clock?.source_window?.first)} — {sourceTime(clock?.source_window?.last)}
        </span>
        <button onClick={onToggleSystem} aria-pressed={showSystem}>
          {showSystem ? "Скрыть панель системы" : "Панель системы"}
        </button>
      </div>
    </footer>
  );
}
