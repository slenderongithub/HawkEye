/**
 * App title bar: the HawkEye wordmark + a live connection chip. Shown on the
 * top-level tab screens. The connection chip reflects the WebSocket phase so
 * the operator always knows whether the feed is live.
 */
import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { palette, spacing, typography } from '@/theme';
import { useLive } from '@/stores/live';
import { Chip } from './Chip';

const CONNECTION_META = {
  open: { label: 'Live', color: palette.green, pulse: true },
  connecting: { label: 'Connecting', color: palette.amber, pulse: true },
  closed: { label: 'Offline', color: palette.red, pulse: false },
  idle: { label: 'Idle', color: palette.textTertiary, pulse: false },
} as const;

type Props = {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
};

export function AppHeader({ title, subtitle, right }: Props) {
  const connection = useLive((s) => s.connection);
  const meta = CONNECTION_META[connection];

  return (
    <View style={styles.header}>
      <View style={styles.left}>
        {title ? (
          <Text style={styles.screenTitle}>{title}</Text>
        ) : (
          <Text style={styles.logo}>
            Hawk<Text style={styles.logoAccent}>Eye</Text>
          </Text>
        )}
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
      <View style={styles.right}>
        {right}
        <Chip label={meta.label} dotColor={meta.color} pulse={meta.pulse} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: spacing.md,
  },
  left: { flexShrink: 1 },
  right: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  logo: {
    ...typography.title,
    color: palette.text,
    letterSpacing: 0.3,
  },
  logoAccent: { color: palette.accent },
  screenTitle: {
    ...typography.title,
    color: palette.text,
  },
  subtitle: {
    ...typography.caption,
    color: palette.textSecondary,
    marginTop: 2,
  },
});
