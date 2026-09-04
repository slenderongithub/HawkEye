/** Centered icon + message for empty lists and error states. */
import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { palette, radii, spacing, typography } from '@/theme';

type Props = {
  icon?: keyof typeof Ionicons.glyphMap;
  title: string;
  subtitle?: string;
  tone?: 'muted' | 'alert';
  children?: React.ReactNode;
};

export function EmptyState({ icon = 'ellipse-outline', title, subtitle, tone = 'muted', children }: Props) {
  const accent = tone === 'alert' ? palette.red : palette.textTertiary;
  return (
    <View style={styles.wrap}>
      <View style={[styles.iconWrap, tone === 'alert' && { backgroundColor: palette.redDim }]}>
        <Ionicons name={icon} size={26} color={accent} />
      </View>
      <Text style={styles.title}>{title}</Text>
      {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xxl,
    paddingHorizontal: spacing.xl,
    gap: spacing.sm,
  },
  iconWrap: {
    width: 56,
    height: 56,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    marginBottom: spacing.xs,
  },
  title: {
    ...typography.heading,
    color: palette.text,
    textAlign: 'center',
  },
  subtitle: {
    ...typography.caption,
    color: palette.textSecondary,
    textAlign: 'center',
    maxWidth: 280,
  },
});
