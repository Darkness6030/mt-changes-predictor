/** Polling without overlapping requests, with abort on unmount and backoff on failure. */

import { useEffect, useRef, useState } from "react";

export interface Poll<T> {
  data: T | null;
  error: string | null;
  updatedAt: number | null;
}

export function usePolling<T>(
  load: (signal: AbortSignal) => Promise<T>,
  intervalMs: number,
  enabled = true,
  resetKey: string | null = null,
): Poll<T> {
  const [state, setState] = useState<Poll<T>>({ data: null, error: null, updatedAt: null });
  const loader = useRef(load);
  loader.current = load;

  useEffect(() => {
    setState({ data: null, error: null, updatedAt: null });
    if (!enabled) return;
    // resetKey restarts the loop immediately, e.g. when another vehicle is selected.
    let cancelled = false;
    let timer: number | undefined;
    let failures = 0;
    const controller = new AbortController();

    const run = async () => {
      try {
        const data = await loader.current(controller.signal);
        if (cancelled) return;
        failures = 0;
        // The previous state is kept on failure: a lost connection must not repaint red as green.
        setState({ data, error: null, updatedAt: Date.now() });
      } catch (error) {
        if (cancelled || (error instanceof DOMException && error.name === "AbortError")) return;
        failures += 1;
        setState((previous) => ({
          ...previous,
          error: error instanceof Error ? error.message : String(error),
        }));
      }
      if (cancelled) return;
      const backoff = Math.min(8, 2 ** failures);
      timer = window.setTimeout(run, intervalMs * (failures ? backoff : 1));
    };

    run();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [intervalMs, enabled, resetKey]);

  return state;
}
