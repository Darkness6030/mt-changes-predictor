import { useState } from "react";
import type { DemoSource, DemoState } from "../types";

const LABELS: Record<DemoSource, string> = {
  replay: "CSV replay", ndtp_replay: "Наш NDTP-replayer", emulator: "Эмулятор организаторов",
};

interface Props {
  state: DemoState | null;
  pending: boolean;
  error: string | null;
  onSwitch: (source: DemoSource, speed: number) => Promise<void>;
}

/** Local demo controls. Selected source always comes from the server, never optimism. */
export function DemoPanel({ state, pending, error, onSwitch }: Props) {
  const [speed, setSpeed] = useState(60);
  if (!state?.enabled) return null;
  const active = state.sources.find((source) => source.id === state.active);
  const unavailable = state.sources.find((source) => !source.available);
  const busy = pending || state.busy;
  const phase = busy ? "Переключение источника…" : state.phase === "completed"
    ? "Отправка завершена" : state.phase === "error" ? "Ошибка источника" : "Источник запущен";
  return <section className="demo-panel" aria-label="Панель демо и отладки">
    <div className="demo-toolbar">
      <strong>Демо / отладка</strong>
      <label>Источник
        <select aria-label="Источник данных" value={state.active} disabled={busy}
          onChange={(event) => void onSwitch(event.target.value as DemoSource, speed)}>
          {state.active === "external_ndtp" ? <option value="external_ndtp">Внешний NDTP</option> : null}
          {state.sources.map((source) => <option key={source.id} value={source.id} disabled={!source.available}>
            {LABELS[source.id]}{!source.available ? " · недоступен" : ""}
          </option>)}
        </select>
      </label>
      {state.active === "ndtp_replay" ? <label>Скорость отправки
        <select aria-label="Скорость нового NDTP-прогона" value={speed} disabled={busy}
          onChange={(event) => setSpeed(Number(event.target.value))}>
          {[1, 10, 30, 60, 120, 300].map((value) => <option key={value} value={value}>{value}×</option>)}
        </select>
      </label> : null}
      <button disabled={busy || !active?.available} title="Очистить состояние и запустить выбранный источник с начала"
        onClick={() => active && void onSwitch(active.id, speed)}>Новый прогон</button>
      <span className="hint" role="status">{phase}{state.active === "ndtp_replay" ? ` · ${state.speed}×` : ""}</span>
      <details className="demo-help"><summary>Об источниках</summary>
        <div>{state.sources.map((source) => <p key={source.id}><b>{LABELS[source.id]}</b>: {source.note}
          {!source.available ? <> {source.reason}</> : null}</p>)}
          <p>Переключение очищает ТС, прогнозы и предупреждения. Перемотка доступна в CSV replay.</p>
        </div>
      </details>
    </div>
    <div className="demo-note">{error ?? state.last_error ?? active?.note}
      {unavailable && !error && !state.last_error ? <span> · {unavailable.reason}</span> : null}
    </div>
  </section>;
}
