/** Header status chip: "KEY  value" with an optional leading dot/icon. */
import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { palette, radii, spacing, typography } from '@/theme';
import { StatusDot } from './StatusDot';

type Props = {
  label: string;
  value?: string | number;
  dotColor?: string;
  pulse?: boolean;
  icon?: keyof typeof Ionicons.glyphMap;
};

export function Chip({ label, value, dotColor, pulse, icon }: Props) {
  return (
    <View style={styles.chip}>
      {dotColor ? <StatusDot color={dotColor} pulse={pulse} size={7} /> : null}
      {icon ? <Ionicons name={icon} size={13} color={palette.textSecondary} /> : null}
      <Text style={styles.label}>{label}</Text>
      {value !== undefined ? <Text style={styles.value}>{value}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.md,
    paddingVertical: 6,
    borderRadius: radii.pill,
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  label: {
    ...typography.caption,
    color: palette.textSecondary,
  },
  value: {
    ...typography.caption,
    color: palette.text,
    fontWeight: '700',
  },
});
