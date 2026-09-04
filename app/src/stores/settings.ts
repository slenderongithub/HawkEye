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

/** Fallback when the origin can't be used (native builds, SSR, file://). */
const FALLBACK_BASE = 'http://127.0.0.1:8000';

/**
 * Default server address.
 *
 * The FastAPI backend serves this app's web build itself, so on web the server
 * is simply wherever the page came from — one address for the UI, the REST API
 * and the WebSocket. Reading `location.origin` means opening the backend just
 * works, with no trip through Settings.
 *
 * Native builds have no origin (the bundle is local), so they keep the LAN
 * default and the Settings override — a phone pointing at a machine on the
 * network is still a real deployment.
 */
function defaultBase(): string {
  if (Platform.OS === 'web') {
    const origin = globalThis.location?.origin;
    // file:// pages report an "null"/opaque origin, which is not a server.
    if (origin && /^https?:/i.test(origin)) return origin.replace(/\/+$/, '');
  }
  return FALLBACK_BASE;
}

const DEFAULT_BASE = defaultBase();

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
