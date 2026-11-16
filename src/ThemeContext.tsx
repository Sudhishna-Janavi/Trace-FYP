/**
 * Which palette the application is drawing in.
 *
 * Trace on the web offers light, dark, or whatever the device prefers, and
 * stores the choice against the account. The same three options exist here,
 * and the choice is sent back to the same endpoint, so changing the theme on
 * the phone changes it in the browser too. It is one account.
 */
import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react';
import { useColorScheme } from 'react-native';

import { api } from './api';
import { Palette, Scheme, palettes } from './theme';

type Preference = 'light' | 'dark' | 'system';

interface ThemeValue {
  p: Palette;
  scheme: Scheme;
  preference: Preference;
  setPreference: (next: Preference) => void;
}

const ThemeContext = createContext<ThemeValue | null>(null);

export function ThemeProvider({ children, initial = 'system' }: {
  children: React.ReactNode;
  initial?: string;
}) {
  const device = useColorScheme();
  const [preference, setLocal] = useState<Preference>(
    initial === 'light' || initial === 'dark' ? initial : 'system');

  useEffect(() => {
    if (initial === 'light' || initial === 'dark' || initial === 'system') {
      setLocal(initial);
    }
  }, [initial]);

  const scheme: Scheme =
    preference === 'system' ? (device === 'dark' ? 'dark' : 'light') : preference;

  const setPreference = useCallback((next: Preference) => {
    setLocal(next);
    // The server is told, but the interface does not wait for it: a theme
    // that only changes once the network answers feels broken.
    api.setTheme(next).catch(() => undefined);
  }, []);

  const value = useMemo<ThemeValue>(
    () => ({ p: palettes[scheme], scheme, preference, setPreference }),
    [scheme, preference, setPreference]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeValue {
  const value = useContext(ThemeContext);
  if (!value) throw new Error('useTheme used outside ThemeProvider');
  return value;
}
