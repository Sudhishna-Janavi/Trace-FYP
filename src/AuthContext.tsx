/**
 * Who is signed in, and what the rest of the application needs to know
 * about them.
 *
 * The token lives in the keychain, so a session survives the application
 * being closed. On launch the token is exchanged for the account it belongs
 * to; if the server rejects it — expired, or signed with a key that has since
 * been rotated — it is discarded and the sign-in screen appears. The
 * application never decides for itself whether a token is still good.
 */
import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react';

import { Me, User, api, clearToken, getToken, isUnauthorised, setToken } from './api';

interface AuthValue {
  ready: boolean;
  user: User | null;
  hasData: boolean;
  reviewCount: number;
  categories: string[];
  threshold: number;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (name: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refresh: () => Promise<void>;
  /** Called after an import, so the shell stops offering to connect. */
  markConnected: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [me, setMe] = useState<Me | null>(null);

  const adopt = useCallback((payload: Me) => {
    setMe(payload);
    setUser(payload.user);
  }, []);

  const refresh = useCallback(async () => {
    try {
      adopt(await api.me());
    } catch (error) {
      if (isUnauthorised(error)) {
        await clearToken();
        setUser(null);
        setMe(null);
      }
      // Any other failure is the network, not the session. Keeping the user
      // signed in means a dropped connection does not throw them out.
    }
  }, [adopt]);

  useEffect(() => {
    (async () => {
      const token = await getToken();
      if (token) await refresh();
      setReady(true);
    })();
  }, [refresh]);

  const signIn = useCallback(async (email: string, password: string) => {
    const session = await api.signIn(email, password);
    await setToken(session.token);
    setUser(session.user);
    await refresh();
  }, [refresh]);

  const signUp = useCallback(async (name: string, email: string, password: string) => {
    const session = await api.signUp(name, email, password);
    await setToken(session.token);
    setUser(session.user);
    await refresh();
  }, [refresh]);

  const signOut = useCallback(async () => {
    await clearToken();
    setUser(null);
    setMe(null);
  }, []);

  const markConnected = useCallback(() => {
    setMe(current => (current ? { ...current, has_data: true } : current));
    refresh();
  }, [refresh]);

  const value = useMemo<AuthValue>(() => ({
    ready,
    user,
    hasData: me?.has_data ?? false,
    reviewCount: me?.review_count ?? 0,
    categories: me?.categories ?? [],
    threshold: me?.threshold ?? 0.6,
    signIn, signUp, signOut, refresh, markConnected,
  }), [ready, user, me, signIn, signUp, signOut, refresh, markConnected]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error('useAuth used outside AuthProvider');
  return value;
}
