/**
 * Connection settings — where the app finds the HawkEye server. Persisted so
 * the operator only enters their server address once. On native we use
 * expo-secure-store; on web we fall back to localStorage (secure-store is a
 * no-op there). Kept deliberately small: host+port is the only thing a Phase 1
 * (LAN, no-auth) client needs.
 */
import { Platform } from 'react-native';
import { create } from 'zustand';
import * as SecureStore from 'expo-secure-store';

const KEY = 'hawkeye.serverBase';

/** Sensible LAN default; the emulator/simulator and web all reach localhost. */
const DEFAULT_BASE = 'http://127.0.0.1:8000';

type SettingsState = {
  /** Base URL, e.g. "http://192.168.1.42:8000" (no trailing slash). */
  serverBase: string;
  hydrated: boolean;
  setServerBase: (base: string) => Promise<void>;
  hydrate: () => Promise<void>;
};

function normalize(base: string): string {
  let b = base.trim();
  if (!b) return DEFAULT_BASE;
  if (!/^https?:\/\//i.test(b)) b = `http://${b}`;
  return b.replace(/\/+$/, '');
}

async function persist(base: string) {
  try {
    if (Platform.OS === 'web') {
      globalThis.localStorage?.setItem(KEY, base);
    } else {
      await SecureStore.setItemAsync(KEY, base);
    }
  } catch {
    // Persistence is best-effort; an in-memory value still works this session.
  }
}

async function load(): Promise<string | null> {
  try {
    if (Platform.OS === 'web') return globalThis.localStorage?.getItem(KEY) ?? null;
    return await SecureStore.getItemAsync(KEY);
  } catch {
    return null;
  }
}

export const useSettings = create<SettingsState>((set) => ({
  serverBase: DEFAULT_BASE,
  hydrated: false,
  setServerBase: async (base) => {
    const clean = normalize(base);
    set({ serverBase: clean });
    await persist(clean);
  },
  hydrate: async () => {
    const stored = await load();
    set({ serverBase: stored ? normalize(stored) : DEFAULT_BASE, hydrated: true });
  },
}));

export { DEFAULT_BASE, normalize as normalizeServerBase };
