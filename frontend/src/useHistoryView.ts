import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { sourceMs } from "./replayTime";
import type { HistoryFrame, Snapshot } from "./types";

interface Cursor { runId: string; at: string }
interface Result { cursor: Cursor; frame: HistoryFrame | null; error: string | null }

/** A browser-local cursor. Live polling and the source clock never pause for a seek. */
export function useHistoryView(live: Snapshot | null) {
  const [cursor, setCursor] = useState<Cursor | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const runId = live?.run_id;
  const active = live?.mode === "ndtp" && cursor?.runId === runId;
  const atMs = sourceMs(cursor?.at);
  const lastMs = sourceMs(live?.history?.last);
  // Retry once when a previously future point enters the archive, not on every head tick.
  const coverage = !Number.isFinite(lastMs) ? "waiting" : atMs > lastMs ? "future" : "arrived";
  useEffect(() => { setCursor(null); setResult(null); }, [runId]);
  useEffect(() => {
    if (!cursor || cursor.runId !== runId) return;
    const controller = new AbortController();
    api.history(cursor.runId, cursor.at, controller.signal).then(
      (frame) => {
        if (!controller.signal.aborted) setResult({ cursor, frame, error: null });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setResult({ cursor, frame: null,
          error: error instanceof Error ? error.message : String(error) });
      },
    );
    return () => controller.abort();
  }, [cursor, runId, coverage]);

  const select = useCallback((at: string) => {
    if (runId) setCursor({ runId, at });
  }, [runId]);
  const follow = useCallback(() => { setCursor(null); setResult(null); }, []);
  const loaded = active && result?.cursor === cursor ? result : null;
  const expired = loaded?.frame && sourceMs(loaded.frame.recorded_at) < sourceMs(live?.history?.first);
  return {
    active: !!active,
    at: active ? cursor!.at : null,
    frame: expired ? null : loaded?.frame ?? null,
    error: expired ? "Выбранная запись уже удалена из журнала. Выберите более позднее время" : loaded?.error ?? null,
    loading: !!active && !loaded,
    select,
    follow,
  };
}
