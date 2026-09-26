import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type ReplayAction } from "./api";
import { Header } from "./components/Header";
import { MapView } from "./components/MapView";
import { Queue, type Filter } from "./components/Queue";
import { ReplayControls } from "./components/ReplayControls";
import { SystemPanel } from "./components/SystemPanel";
import { VehicleCard } from "./components/VehicleCard";
import { usePolling } from "./usePolling";
import type { Snapshot, Status, VehicleDetail } from "./types";

export default function App() {
  const [filter, setFilter] = useState<Filter>("attention");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [showSystem, setShowSystem] = useState(false);
  const [commandPending, setCommandPending] = useState(false);
  const [pollKey, setPollKey] = useState(0);
  const [commandError, setCommandError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [lastStatus, setLastStatus] = useState<Status | null>(null);

  const snapshotPoll = usePolling((signal) => api.snapshot(signal), 1000, true, String(pollKey));
  const statusPoll = usePolling((signal) => api.status(signal), 4000, true, String(pollKey));
  const qualityPoll = usePolling((signal) => api.quality(signal), 10000, showSystem);
  const detailPoll = usePolling<VehicleDetail>(
    (signal) => api.vehicle(selected as string, signal),
    2000,
    selected !== null,
    `${snapshot?.run_id ?? ""}:${selected ?? ""}`,
  );

  useEffect(() => { if (statusPoll.data) setLastStatus(statusPoll.data); }, [statusPoll.data]);
  const navigationStatus = statusPoll.data ?? lastStatus;

  // A stale answer must never overwrite a newer snapshot revision.
  useEffect(() => {
    const fresh = snapshotPoll.data;
    if (!fresh) return;
    if (fresh.run_id !== snapshot?.run_id || fresh.revision >= revision) {
      setSnapshot(fresh);
      setRevision(fresh.revision);
    }
  }, [snapshotPoll.data, snapshot?.run_id, revision]);

  useEffect(() => { setSelected(null); }, [snapshot?.run_id]);

  const detail = useMemo(() => {
    const current = snapshot?.vehicles.find((item) => item.tr_id === selected);
    if (!current || !snapshot) return null;
    const loaded = detailPoll.data;
    const usable = loaded?.tr_id === selected && loaded?.run_id === snapshot.run_id && !detailPoll.error;
    if (usable) {
      return { ...loaded, ...(loaded.revision > snapshot.revision ? {} : current) };
    }
    return { ...current, run_id: snapshot.run_id, revision: snapshot.revision,
      track: [], plan: [], prediction_history: [] } as VehicleDetail;
  }, [detailPoll.data, detailPoll.error, snapshot, selected]);

  const acknowledge = useCallback(async (alertId: string) => {
    try {
      await api.acknowledge(alertId);
      setCommandError(null);
    } catch (error) {
      setCommandError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const command = useCallback(
    async (action: ReplayAction, speed?: number, startAt?: string) => {
      setCommandPending(true);
      try {
        await api.replay(action, speed, startAt);
        if (action === "seek" || action === "reset") setSelected(null);
        setPollKey((value) => value + 1);
        setCommandError(null);
      } catch (error) {
        setCommandError(error instanceof Error ? error.message : String(error));
      } finally {
        setCommandPending(false);
      }
    },
    [],
  );

  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const ageMs = snapshotPoll.updatedAt ? Math.max(0, now - snapshotPoll.updatedAt) : null;
  const connectionLost = snapshotPoll.error !== null || (ageMs !== null && ageMs > 6000);
  const clock = snapshot?.clock;

  return (
    <div className="app">
      <Header snapshot={snapshot} status={statusPoll.data} ageMs={ageMs} />
      {connectionLost ? (
        <div className="banner">
          Связь с Backend потеряна: {snapshotPoll.error ?? "состояние давно не обновлялось"}. Показано последнее полученное состояние
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
      {selected && detailPoll.error ? <div className="banner warn">
        История ТС недоступна: {detailPoll.error}. Основные данные обновляются из общей очереди.
      </div> : null}
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
        status={navigationStatus ? { ...navigationStatus, clock: snapshot?.run_id === navigationStatus.run_id ? snapshot.clock : navigationStatus.clock } : null}
        pending={commandPending}
        onCommand={command}
        showSystem={showSystem}
        onToggleSystem={() => setShowSystem((value) => !value)}
        error={commandError}
      />
    </div>
  );
}
