/**
 * Live state fed by the WebSocket: the most recent video frame (as an <Image>
 * source uri), the latest runtime status, and the connection phase. Screens
 * subscribe to just the slice they need so a 15fps frame update doesn't
 * re-render the whole tree.
 */
import { create } from 'zustand';
import type { Status } from '@/api/types';

export type ConnectionPhase = 'idle' | 'connecting' | 'open' | 'closed';

type LiveState = {
  frameUri: string | null;
  status: Status | null;
  connection: ConnectionPhase;
  lastFrameAt: number | null;
  setFrame: (uri: string) => void;
  setStatus: (status: Status) => void;
  setConnection: (phase: ConnectionPhase) => void;
  reset: () => void;
};

export const useLive = create<LiveState>((set) => ({
  frameUri: null,
  status: null,
  connection: 'idle',
  lastFrameAt: null,
  setFrame: (uri) => set({ frameUri: uri, lastFrameAt: Date.now() }),
  setStatus: (status) => set({ status }),
  setConnection: (connection) => set({ connection }),
  reset: () => set({ frameUri: null, status: null, connection: 'idle', lastFrameAt: null }),
}));
