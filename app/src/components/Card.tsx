/** Base surface container. Optional press behavior + accent-tinted variant. */
import React from 'react';
import { View, StyleSheet, type ViewStyle, type StyleProp } from 'react-native';
import { PressableScale } from './PressableScale';
import { palette, radii, spacing } from '@/theme';

type Props = {
  children: React.ReactNode;
  onPress?: () => void;
  style?: StyleProp<ViewStyle>;
  /**
   * Elevated variant: a micro-step lighter surface plus a stronger hairline.
   * Deliberately not a shadow — the design language keeps its single drop-shadow
   * for the live feed and builds every other level from surface colour.
   */
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
    borderColor: palette.hairlineStrong,
  },
  padded: {
    padding: spacing.lg,
  },
});
