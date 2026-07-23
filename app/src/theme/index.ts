import { palette, type Palette } from './tokens';
import type { Severity, IncidentStatus, ActionType, AnomalyType } from '@/api/types';

export * from './tokens';

/** Severity → color pair (text/foreground + dim background). */
export function severityColors(sev: Severity | string | undefined): {
  fg: string;
  bg: string;
  solid: boolean;
} {
  switch (sev) {
    case 'critical':
      return { fg: palette.white, bg: palette.redSolid, solid: true };
    case 'high':
      return { fg: palette.red, bg: palette.redDim, solid: false };
    case 'medium':
      return { fg: palette.amber, bg: palette.amberDim, solid: false };
    case 'low':
    default:
      return { fg: palette.green, bg: palette.greenDim, solid: false };
  }
}

/** Incident status → color. */
export function statusColors(status: IncidentStatus | string | undefined): {
  fg: string;
  bg: string;
} {
  switch (status) {
    case 'confirmed':
      return { fg: palette.red, bg: palette.redDim };
    case 'dismissed':
      return { fg: palette.textSecondary, bg: palette.surfaceElevated };
    case 'detected':
    default:
      return { fg: palette.accent, bg: palette.accentDim };
  }
}

/** Dispatch action → color. */
export function actionColors(action: ActionType | string | undefined, dispatched?: boolean): {
  fg: string;
  bg: string;
} {
  if (dispatched) return { fg: palette.red, bg: palette.redDim };
  if (!action || action === 'none') return { fg: palette.textSecondary, bg: palette.surfaceElevated };
  return { fg: palette.accent, bg: palette.accentDim };
}

/**
 * Anomaly type → Ionicons glyph + accent color. Drives icons across the app
 * (incident cards, live overlays, detail headers) so a given anomaly always
 * looks the same. Keys mirror the backend AnomalyType enum.
 */
export const anomalyMeta: Record<
  string,
  { icon: string; color: keyof Palette; label: string }
> = {
  normal: { icon: 'checkmark-circle', color: 'green', label: 'Normal' },
  violence: { icon: 'hand-left', color: 'red', label: 'Violence' },
  robbery: { icon: 'cash', color: 'red', label: 'Robbery' },
  theft: { icon: 'bag-remove', color: 'amber', label: 'Theft' },
  weapon: { icon: 'alert', color: 'red', label: 'Weapon' },
  intrusion: { icon: 'log-in', color: 'amber', label: 'Intrusion' },
  fall: { icon: 'body', color: 'amber', label: 'Fall' },
  medical: { icon: 'medkit', color: 'purple', label: 'Medical' },
  accident: { icon: 'warning', color: 'amber', label: 'Accident' },
  fire: { icon: 'flame', color: 'red', label: 'Fire' },
  unknown: { icon: 'help-circle', color: 'textSecondary', label: 'Unknown' },
};

export function anomalyInfo(type: AnomalyType | string | undefined) {
  return anomalyMeta[type ?? 'unknown'] ?? anomalyMeta.unknown;
}
