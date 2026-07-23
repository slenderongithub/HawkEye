/**
 * Thin typed fetch wrapper around the HawkEye REST API. Reads the base URL from
 * the settings store at call time (so changing the server address takes effect
 * without a reload) and validates every response against its zod schema.
 */
import { z } from 'zod';
import { useSettings } from '@/stores/settings';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function base(): string {
  return useSettings.getState().serverBase;
}

/** Absolute URL for a REST path (path should start with "/"). */
export function apiUrl(path: string): string {
  return `${base()}${path}`;
}

/** ws:// or wss:// URL for the given path, derived from the http(s) base. */
export function wsUrl(path: string): string {
  return `${base().replace(/^http/i, 'ws')}${path}`;
}

// Infer the schema's *output* type (post-parse, with defaults/catch applied),
// which differs from its input type for schemas that use .default()/.catch().
async function request<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  init?: RequestInit & { timeoutMs?: number },
): Promise<z.output<S>> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), init?.timeoutMs ?? 12_000);
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { ...init, signal: controller.signal });
  } catch (e) {
    throw new ApiError(
      e instanceof Error && e.name === 'AbortError'
        ? 'Request timed out — is the server reachable?'
        : 'Cannot reach the HawkEye server. Check the address in Settings.',
      0,
    );
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(`Request failed (${res.status})`, res.status, text);
  }

  const json = await res.json();
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    throw new ApiError('Server returned an unexpected response shape.', res.status);
  }
  return parsed.data;
}

export const api = {
  get: <S extends z.ZodTypeAny>(path: string, schema: S, init?: RequestInit & { timeoutMs?: number }) =>
    request(path, schema, { ...init, method: 'GET' }),

  /** POST multipart/form-data (used by enrollment). Returns the parsed body. */
  postForm: <S extends z.ZodTypeAny>(path: string, form: FormData, schema: S) =>
    request(path, schema, { method: 'POST', body: form, timeoutMs: 30_000 }),

  del: <S extends z.ZodTypeAny>(path: string, schema: S) =>
    request(path, schema, { method: 'DELETE' }),
};
