import { useEffect, useRef, useState } from "react";
import type { DemoSource, DemoState } from "../types";

const LABELS: Record<DemoSource, string> = {
  replay: "CSV replay", ndtp_replay: "Наш NDTP-replayer", emulator: "Эмулятор организаторов",
};
const SHORT_LABELS: Record<DemoSource, string> = {
  replay: "CSV replay", ndtp_replay: "NDTP replay", emulator: "Эмулятор",
};

interface Props {
  state: DemoState | null;
  pending: boolean;
  error: string | null;
  onSwitch: (source: DemoSource, speed: number) => Promise<void>;
}

/** Source controls share the time toolbar; infrequent actions live in an anchored popover. */
export function DemoPanel({ state, pending, error, onSwitch }: Props) {
  const [speed, setSpeed] = useState(60);
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);

  if (!state?.enabled) return null;
  const active = state.sources.find((source) => source.id === state.active);
  const busy = pending || state.busy;
  const label = state.active === "external_ndtp" ? "Внешний NDTP" : SHORT_LABELS[state.active];
  const issue = error ?? state.last_error;
  const phase = busy ? "Переключение источника…" : state.phase === "completed"
    ? "Отправка завершена" : state.phase === "error" ? "Ошибка источника" : "Источник запущен";
  const close = () => { setOpen(false); trigger.current?.focus(); };
  const change = (source: DemoSource) => {
    close();
    void onSwitch(source, speed);
  };

  return <div className="demo-source" ref={root} onKeyDown={(event) => {
    if (event.key === "Escape") { event.stopPropagation(); close(); }
  }}>
    <button ref={trigger} className={`source-trigger${issue ? " has-issue" : ""}`}
      aria-label={`Источник данных: ${label}`} aria-haspopup="dialog" aria-expanded={open}
      aria-controls={open ? "demo-source-dialog" : undefined} disabled={busy}
      title={busy ? phase : `Источник данных: ${label}. Настройки демо и отладки`}
      onClick={() => setOpen((value) => !value)}>
      <span className={`source-indicator${busy ? " busy" : issue ? " error" : ""}`} />
      {label}<span className="chevron">⌄</span>
    </button>
    {open ? <section id="demo-source-dialog" className="source-popover" role="dialog" aria-label="Демо и источники данных">
      <div className="time-editor-heading"><strong>Источник данных</strong>
        <button ref={closeButton} aria-label="Закрыть выбор источника" onClick={close}>×</button></div>
      <div className="source-options" role="group" aria-label="Выбор источника">
        {state.sources.map((source) => <button key={source.id} className="source-option"
          aria-pressed={source.id === state.active} disabled={busy || !source.available}
          onClick={() => source.id === state.active ? close() : change(source.id)}>
          <span className="source-option-title"><b>{LABELS[source.id]}</b>
            <span>{source.id === state.active ? "✓" : !source.available ? "Недоступен" : ""}</span></span>
          <span className="source-description">{source.note}</span>
          {!source.available ? <span className="source-description">{source.reason}</span> : null}
        </button>)}
      </div>
      <div className="source-run-controls">
        {state.active === "ndtp_replay" ? <label>Скорость нового прогона
          <select aria-label="Скорость нового NDTP-прогона" value={speed} disabled={busy}
            onChange={(event) => setSpeed(Number(event.target.value))}>
            {[1, 10, 30, 60, 120, 300].map((value) => <option key={value} value={value}>{value}×</option>)}
          </select>
        </label> : null}
        <button disabled={busy || !active?.available} onClick={() => active && change(active.id)}>Новый прогон</button>
        <span className="hint" role="status">{phase}{state.active === "ndtp_replay" ? ` · ${state.speed}×` : ""}</span>
      </div>
      {issue ? <p className="source-error" role="alert">{issue}</p> : null}
      <p className="source-footnote">Переключение очищает ТС, прогнозы и предупреждения.</p>
    </section> : null}
  </div>;
}
