/**
 * react-query hooks over the REST endpoints. Query keys are namespaced by the
 * server base so switching servers doesn't show a stale cache. The live
 * WebSocket (useLiveSocket) is the real-time channel; these cover history +
 * mutations (list/enroll/delete people, incident/attendance history).
 */
import { z } from 'zod';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import { useSettings } from '@/stores/settings';
import {
  AttendanceRecord,
  Incident,
  Person,
  Status,
  Health,
} from './types';
// Names above are zod schemas *and* their inferred types (declaration merge),
// so `Incident`, `Person`, etc. below are used as TS types.
type IncidentT = Incident;
type AttendanceT = AttendanceRecord;
type PersonT = Person;
type StatusT = Status;
type HealthT = Health;

const IncidentList = z.array(Incident);
const AttendanceList = z.array(AttendanceRecord);
const PeopleList = z.array(Person);
const DeleteResult = z.object({ ok: z.boolean() });

function useBaseKey() {
  return useSettings((s) => s.serverBase);
}

/**
 * Shared default incident-list size. All incident consumers (Live preview,
 * Incidents feed, and the detail screen's cache lookup) use this so they share
 * ONE react-query cache entry (['incidents', base, INCIDENTS_LIMIT]) — tapping
 * a Live incident resolves from cache instead of flashing a skeleton + refetch.
 */
export const INCIDENTS_LIMIT = 50;

export function useIncidents(limit = INCIDENTS_LIMIT) {
  const base = useBaseKey();
  return useQuery<IncidentT[]>({
    queryKey: ['incidents', base, limit],
    queryFn: () => api.get(`/api/incidents?limit=${limit}`, IncidentList),
    staleTime: 15_000,
  });
}

export function useAttendance(date?: string) {
  const base = useBaseKey();
  return useQuery<AttendanceT[]>({
    queryKey: ['attendance', base, date ?? 'today'],
    queryFn: () => api.get(`/api/attendance${date ? `?date=${date}` : ''}`, AttendanceList),
    staleTime: 10_000,
  });
}

export function usePeople() {
  const base = useBaseKey();
  return useQuery<PersonT[]>({
    queryKey: ['people', base],
    queryFn: () => api.get('/api/people', PeopleList),
    staleTime: 30_000,
  });
}

export function useStatus() {
  const base = useBaseKey();
  return useQuery<StatusT>({
    queryKey: ['status', base],
    queryFn: () => api.get('/api/status', Status),
    refetchInterval: 5_000,
  });
}

export function useHealth() {
  const base = useBaseKey();
  return useQuery<HealthT | null>({
    queryKey: ['health', base],
    // /api/health returns 503 when degraded — surface the body either way.
    queryFn: () => api.get('/api/health', Health).catch(() => null),
    refetchInterval: 10_000,
    retry: false,
  });
}

/** Multipart enrollment: name + optional meta JSON + one-or-more images. */
export function useEnrollPerson() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (input: {
      name: string;
      meta?: Record<string, string>;
      images: { uri: string; name?: string; type?: string }[];
    }) => {
      const form = new FormData();
      form.append('name', input.name);
      form.append('meta', JSON.stringify(input.meta ?? {}));
      input.images.forEach((img, i) => {
        // RN FormData file shape; on web, callers pass a Blob-backed uri.
        form.append('files', {
          uri: img.uri,
          name: img.name ?? `enroll_${i}.jpg`,
          type: img.type ?? 'image/jpeg',
        } as unknown as Blob);
      });
      return api.postForm('/api/people', form, Person);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['people'] }),
  });
}

export function useDeletePerson() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.del(`/api/people/${id}`, DeleteResult),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['people'] }),
  });
}
