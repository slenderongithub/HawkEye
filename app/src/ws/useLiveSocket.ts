/**
 * Single WebSocket connection to the server's /ws endpoint.
 *
 * Protocol (server/app.py):
 *   - binary message  → a JPEG video frame (~15fps)
 *   - text message    → JSON {type: "incident"|"attendance"|"status", data}
 *
 * Frames become an <Image> source uri:
 *   - web    → object URL (cheap; revoked on replace)
 *   - native → base64 data URI (Hermes has atob/btoa)
 *
 * Events are validated with zod and fanned out: status → live store, incidents
 * & attendance → react-query cache (so history screens stay live) + toasts.
 * Auto-reconnects with backoff. Mount this once, high in the tree (root layout).
 */
import { useEffect, useRef } from 'react';
import { Platform } from 'react-native';
import { useQueryClient } from '@tanstack/react-query';
import { wsUrl } from '@/api/client';
import { WsEvent, type Incident, type AttendanceRecord } from '@/api/types';
import { useLive } from '@/stores/live';
import { useSettings } from '@/stores/settings';
import { toast } from '@/stores/toast';
import { anomalyInfo } from '@/theme';

/** Chunked ArrayBuffer → base64 (avoids call-stack blowups on large frames). */
function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(
      null,
      bytes.subarray(i, i + CHUNK) as unknown as number[],
    );
  }
  // btoa exists on web and in Hermes (RN 0.74+).
  return globalThis.btoa(binary);
}

export function useLiveSocket() {
  const serverBase = useSettings((s) => s.serverBase);
  const hydrated = useSettings((s) => s.hydrated);
  const qc = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const lastObjectUrl = useRef<string | null>(null);
  const retry = useRef(0);
  const closedByUs = useRef(false);

  useEffect(() => {
    if (!hydrated) return;
    closedByUs.current = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const { setFrame, setStatus, setConnection } = useLive.getState();

    const connect = () => {
      setConnection('connecting');
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl('/ws'));
      } catch {
        scheduleReconnect();
        return;
      }
      socket.binaryType = Platform.OS === 'web' ? 'blob' : 'arraybuffer';
      wsRef.current = socket;

      socket.onopen = () => {
        retry.current = 0;
        setConnection('open');
      };

      socket.onmessage = (ev: WebSocketMessageEvent) => {
        const data = ev.data;
        // --- binary: a video frame ---
        if (Platform.OS === 'web' && data instanceof Blob) {
          const url = URL.createObjectURL(data);
          if (lastObjectUrl.current) URL.revokeObjectURL(lastObjectUrl.current);
          lastObjectUrl.current = url;
          setFrame(url);
          return;
        }
        if (data instanceof ArrayBuffer) {
          setFrame(`data:image/jpeg;base64,${arrayBufferToBase64(data)}`);
          return;
        }
        // --- text: a JSON event ---
        if (typeof data === 'string') {
          handleEvent(data);
        }
      };

      socket.onclose = () => {
        setConnection('closed');
        if (!closedByUs.current) scheduleReconnect();
      };
      socket.onerror = () => {
        // onclose will follow and drive reconnect.
      };
    };

    const handleEvent = (raw: string) => {
      let json: unknown;
      try {
        json = JSON.parse(raw);
      } catch {
        return;
      }
      const parsed = WsEvent.safeParse(json);
      if (!parsed.success) return;
      const evt = parsed.data;

      if (evt.type === 'status') {
        setStatus(evt.data);
      } else if (evt.type === 'incident') {
        pushIncident(evt.data);
      } else if (evt.type === 'attendance') {
        pushAttendance(evt.data);
      }
    };

    const pushIncident = (inc: Incident) => {
      // Prepend into every cached incidents list (keyed by base + limit).
      qc.setQueriesData<Incident[]>({ queryKey: ['incidents'] }, (old: Incident[] | undefined) => {
        if (!old) return old;
        if (old.some((i: Incident) => i.id === inc.id)) return old;
        return [inc, ...old];
      });
      const sev = inc.reasoning?.severity;
      if (sev === 'high' || sev === 'critical') {
        const info = anomalyInfo(inc.anomaly_type);
        toast.show(inc.headline || info.label, 'alert', info.icon);
      }
    };

    const pushAttendance = (rec: AttendanceRecord) => {
      qc.setQueriesData<AttendanceRecord[]>({ queryKey: ['attendance'] }, (old: AttendanceRecord[] | undefined) => {
        if (!old) return old;
        const idx = old.findIndex((r: AttendanceRecord) => r.person_id === rec.person_id && r.date === rec.date);
        if (idx >= 0) {
          const next = old.slice();
          next[idx] = rec;
          return next;
        }
        return [rec, ...old];
      });
      toast.show(`${rec.name} marked present`, 'success', 'checkmark-circle');
    };

    const scheduleReconnect = () => {
      if (closedByUs.current) return;
      retry.current = Math.min(retry.current + 1, 6);
      const delay = Math.min(1000 * retry.current, 6000);
      reconnectTimer = setTimeout(connect, delay);
    };

    connect();

    return () => {
      closedByUs.current = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (lastObjectUrl.current) {
        URL.revokeObjectURL(lastObjectUrl.current);
        lastObjectUrl.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
      useLive.getState().setConnection('idle');
    };
    // Reconnect from scratch whenever the server address changes.
  }, [serverBase, hydrated, qc]);
}
