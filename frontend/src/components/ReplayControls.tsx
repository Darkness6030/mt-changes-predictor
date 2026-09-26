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
      <span>
        {status?.mode === "ndtp"
          ? "Живой приём NDTP: виртуальные часы следуют за потоком"
          : "Исторический replay по времени источника"}
      </span>
      {replay ? (
        <>
          <button onClick={() => onCommand(clock?.paused ? "start" : "pause")}>
            {clock?.paused ? "Продолжить" : "Пауза"}
          </button>
          {SPEEDS.map((speed) => (
            <button
              key={speed}
              className={clock?.speed === speed ? "active" : ""}
              onClick={() => onCommand("speed", speed)}
            >
              {speed}×
            </button>
          ))}
          <button onClick={() => onCommand("reset")}>Сброс прогона</button>
        </>
      ) : (
        <span className="hint">управление replay недоступно в этом режиме</span>
      )}
      <span className="spacer" />
      {error ? <span className="risk-red">{error}</span> : null}
      <span>
        период {sourceTime(clock?.source_window?.first)} — {sourceTime(clock?.source_window?.last)}
      </span>
      <button onClick={onToggleSystem} aria-pressed={showSystem}>
        {showSystem ? "Скрыть панель системы" : "Панель системы"}
      </button>
    </footer>
  );
}
