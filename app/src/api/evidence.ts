/**
 * Evidence media lives on the server under data/evidence/<hash>/... and is
 * served (path-traversal guarded) by GET /api/evidence/{path}. The backend
 * hands us bare relative paths in Incident.thumb / face_match.face_image /
 * clip_path; this turns them into absolute URLs the app's <Image> can load.
 */
import { apiUrl } from './client';

export function evidenceUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  // Already absolute (defensive — server currently sends relative paths).
  if (/^https?:\/\//i.test(path)) return path;
  return apiUrl(`/api/evidence/${path}`);
}
