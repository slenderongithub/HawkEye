/** Base surface container. Optional press behavior + accent-tinted variant. */
import React from 'react';
import { View, StyleSheet, type ViewStyle, type StyleProp } from 'react-native';
import { PressableScale } from './PressableScale';
import { palette, radii, spacing, shadows } from '@/theme';

type Props = {
  children: React.ReactNode;
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
  /** Elevated variant: brighter surface + soft shadow (for hero cards). */
  elevated?: boolean;
  padded?: boolean;
};

export function Card({ children, onPress, style, elevated, padded }: Props) {
  const content = (
    <View
      style={[
        styles.card,
        elevated && styles.elevated,
        padded && styles.padded,
        style,
      ]}
    >
      {children}
    </View>
  );
  if (onPress) {
    return <PressableScale onPress={onPress}>{content}</PressableScale>;
  }
  return content;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: palette.surface,
    borderRadius: radii.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    overflow: 'hidden',
  },
  elevated: {
    backgroundColor: palette.surfaceElevated,
    ...shadows.card,
  },
  padded: {
    padding: spacing.lg,
  },
});
