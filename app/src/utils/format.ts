/** Small formatting helpers. Timestamps from the backend are epoch seconds. */

export function timeOfDay(tsSeconds: number | null | undefined): string {
  if (!tsSeconds) return '—';
  return new Date(tsSeconds * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function dateLabel(tsSeconds: number | null | undefined): string {
  if (!tsSeconds) return '—';
  return new Date(tsSeconds * 1000).toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
  });
}

/** "just now" / "4m ago" / "2h ago" / "Jul 21". */
export function relativeTime(tsSeconds: number | null | undefined): string {
  if (!tsSeconds) return '—';
  const diffMs = Date.now() - tsSeconds * 1000;
  const sec = Math.round(diffMs / 1000);
  if (sec < 45) return 'just now';
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return dateLabel(tsSeconds);
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value == null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function titleCase(s: string | null | undefined): string {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** "YYYY-MM-DD" in local time — matches AttendanceRecord.date. */
export function isoDate(d: Date = new Date()): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}
