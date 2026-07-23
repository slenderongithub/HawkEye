/**
 * Compact colored label. Two convenience wrappers map the backend enums to the
 * right palette so severity/action always render identically across screens.
 */
import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { palette, radii, spacing, typography, severityColors, actionColors } from '@/theme';
import type { Severity, ActionType } from '@/api/types';

type BadgeProps = {
  label: string;
  fg: string;
  bg: string;
  solid?: boolean;
};

export function Badge({ label, fg, bg }: BadgeProps) {
  return (
    <View style={[styles.badge, { backgroundColor: bg }]}>
      <Text style={[styles.text, { color: fg }]}>{label}</Text>
    </View>
  );
}

export function SeverityBadge({ severity }: { severity: Severity | string | undefined }) {
  const c = severityColors(severity);
  return <Badge label={String(severity ?? 'low')} fg={c.fg} bg={c.bg} />;
}

export function ActionBadge({
  action,
  dispatched,
}: {
  action: ActionType | string | undefined;
  dispatched?: boolean;
}) {
  const c = actionColors(action, dispatched);
  const label = dispatched ? `dispatch: ${action}` : String(action ?? 'none');
  return <Badge label={label} fg={c.fg} bg={c.bg} />;
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 3,
    borderRadius: radii.sm,
    alignSelf: 'flex-start',
  },
  text: {
    ...typography.micro,
    textTransform: 'uppercase',
  },
});
