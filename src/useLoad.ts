/**
 * Fetch something, and keep the four states a screen actually has.
 *
 * Loading, loaded, failed, and refreshing-while-already-showing-something.
 * The fourth is the one that is easy to forget and the one that matters on a
 * phone: pulling to refresh should not blank the screen you were reading.
 *
 * A rejected session is not handled here. It propagates, and the provider
 * that owns the token decides to sign out, so that decision lives in one
 * place rather than in every screen.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError } from './api';

interface State<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  refreshing: boolean;
  reload: () => void;
  refresh: () => void;
  set: (next: T) => void;
}

export function useLoad<T>(fetcher: () => Promise<T>, deps: unknown[] = []): State<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  // A screen can be left while a request is in flight; writing state after
  // that is harmless but noisy, so it is skipped.
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);

  const run = useCallback(async (quiet: boolean) => {
    if (quiet) setRefreshing(true); else setLoading(true);
    setError(null);
    try {
      const result = await fetcher();
      if (alive.current) setData(result);
    } catch (problem) {
      if (alive.current) {
        setError(problem instanceof ApiError
          ? problem.message
          : 'Could not load that.');
      }
    } finally {
      if (alive.current) { setLoading(false); setRefreshing(false); }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => { run(false); }, [run]);

  return {
    data, error, loading, refreshing,
    reload: () => run(false),
    refresh: () => run(true),
    set: setData,
  };
}
