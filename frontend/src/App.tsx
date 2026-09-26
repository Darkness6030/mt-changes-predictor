import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { Header } from "./components/Header";
import { MapView } from "./components/MapView";
import { Queue, type Filter } from "./components/Queue";
import { ReplayControls } from "./components/ReplayControls";
import { SystemPanel } from "./components/SystemPanel";
import { VehicleCard } from "./components/VehicleCard";
import { usePolling } from "./usePolling";
import type { Snapshot, VehicleDetail } from "./types";

export default function App() {
  const [filter, setFilter] = useState<Filter>("attention");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [showSystem, setShowSystem] = useState(false);
  const [commandError, setCommandError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);

  const snapshotPoll = usePolling((signal) => api.snapshot(signal), 1000);
  const statusPoll = usePolling((signal) => api.status(signal), 4000);
  const qualityPoll = usePolling((signal) => api.quality(signal), 10000, showSystem);
  const detailPoll = usePolling<VehicleDetail>(
    (signal) => api.vehicle(selected as string, signal),
    2000,
    selected !== null,
    selected,
  );

  // A stale answer must never overwrite a newer snapshot revision.
  useEffect(() => {
    const fresh = snapshotPoll.data;
    if (!fresh) return;
    if (fresh.run_id !== snapshot?.run_id || fresh.revision >= revision) {
      setSnapshot(fresh);
      setRevision(fresh.revision);
    }
  }, [snapshotPoll.data, snapshot?.run_id, revision]);

  // A new run starts with clean local selection instead of mixing vehicles between runs.
  useEffect(() => {
    if (snapshot && selected && !snapshot.vehicles.some((item) => item.tr_id === selected)) {
      const stillKnown = snapshot.vehicles.length === 0;
      if (!stillKnown) setSelected(null);
    }
  }, [snapshot?.run_id]);

  const detail = useMemo(() => {
    const loaded = detailPoll.data;
    if (loaded && loaded.tr_id === selected) return loaded;
    const fallback = snapshot?.vehicles.find((item) => item.tr_id === selected);
    return fallback ? ({ ...fallback, track: [], plan: [], prediction_history: [] } as VehicleDetail) : null;
  }, [detailPoll.data, snapshot, selected]);

  const acknowledge = useCallback(async (alertId: string) => {
    try {
      await api.acknowledge(alertId);
      setCommandError(null);
    } catch (error) {
      setCommandError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const command = useCallback(
    async (action: "start" | "pause" | "reset" | "speed", speed?: number) => {
      try {
        await api.replay(action, speed);
        setCommandError(null);
      } catch (error) {
        setCommandError(error instanceof Error ? error.message : String(error));
      }
    },
    [],
  );

  const ageMs = snapshotPoll.updatedAt ? Date.now() - snapshotPoll.updatedAt : null;
  const connectionLost = snapshotPoll.error !== null;
  const clock = snapshot?.clock;

  return (
    <div className="app">
      <Header snapshot={snapshot} status={statusPoll.data} ageMs={ageMs} />
      {connectionLost ? (
        <div className="banner">
          Связь с Backend потеряна: {snapshotPoll.error}. Показано последнее полученное состояние
          {ageMs ? ` (${Math.round(ageMs / 1000)} с назад)` : ""}.
        </div>
      ) : null}
      {statusPoll.data && !statusPoll.data.ml.ready ? (
        <div className="banner warn">
          ML-сервис недоступен: новые прогнозы не выдаются, показаны прошлые значения с их
          возрастом. {statusPoll.data.ml.last_error ?? ""}
        </div>
      ) : null}
      {clock?.plan_shift_note ? <div className="banner warn">{clock.plan_shift_note}</div> : null}
      {snapshot?.fixture ? (
        <div className="banner warn">Демонстрационные фикстуры, не живой поток.</div>
      ) : null}
      <div className="body">
        <Queue
          snapshot={snapshot}
          filter={filter}
          search={search}
          selected={selected}
          onFilter={setFilter}
          onSearch={setSearch}
          onSelect={setSelected}
        />
        <MapView snapshot={snapshot} detail={detail} selected={selected} onSelect={setSelected} />
        {showSystem ? (
          <SystemPanel status={statusPoll.data} quality={qualityPoll.data} />
        ) : (
          <VehicleCard
            detail={detail}
            alerts={snapshot?.alerts ?? []}
            policy={snapshot?.risk_policy ?? null}
            onAcknowledge={acknowledge}
          />
        )}
      </div>
      <ReplayControls
        status={statusPoll.data}
        onCommand={command}
        showSystem={showSystem}
        onToggleSystem={() => setShowSystem((value) => !value)}
        error={commandError}
      />
    </div>
  );
}
