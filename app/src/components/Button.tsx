/** Primary / secondary / danger button with spring press + busy state. */
import React from 'react';
import { ActivityIndicator, Text, View, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { PressableScale } from './PressableScale';
import { palette, radii, spacing, typography } from '@/theme';

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost';

type Props = {
  label: string;
  onPress?: () => void;
  variant?: Variant;
  icon?: keyof typeof Ionicons.glyphMap;
  busy?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
};

export function Button({
  label,
  onPress,
  variant = 'primary',
  icon,
  busy,
  disabled,
  fullWidth,
}: Props) {
  const v = VARIANTS[variant];
  const isDisabled = disabled || busy;

  return (
    <PressableScale
      onPress={isDisabled ? undefined : onPress}
      disabled={isDisabled}
      style={[
        styles.base,
        { backgroundColor: v.bg, borderColor: v.border },
        fullWidth && styles.fullWidth,
        isDisabled && styles.disabled,
      ]}
    >
      <View style={styles.row}>
        {busy ? (
          <ActivityIndicator size="small" color={v.fg} />
        ) : icon ? (
          <Ionicons name={icon} size={17} color={v.fg} />
        ) : null}
        <Text style={[styles.label, { color: v.fg }]}>{label}</Text>
      </View>
    </PressableScale>
  );
}

const VARIANTS: Record<Variant, { bg: string; fg: string; border: string }> = {
  primary: { bg: palette.accent, fg: palette.textInverse, border: palette.accent },
  secondary: {
    bg: palette.surfaceElevated,
    fg: palette.text,
    border: palette.hairlineStrong,
  },
  danger: { bg: palette.redDim, fg: palette.red, border: 'transparent' },
  ghost: { bg: 'transparent', fg: palette.textSecondary, border: 'transparent' },
};

const styles = StyleSheet.create({
  base: {
    borderRadius: radii.md,
    borderWidth: StyleSheet.hairlineWidth,
    paddingVertical: 13,
    paddingHorizontal: spacing.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  fullWidth: { width: '100%' },
  disabled: { opacity: 0.5 },
  row: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  label: { ...typography.bodyStrong },
});
