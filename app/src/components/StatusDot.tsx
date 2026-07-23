/** A small status dot with an optional soft glow + breathing pulse. */
import React, { useEffect } from 'react';
import { StyleSheet } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
  Easing,
} from 'react-native-reanimated';
import { palette } from '@/theme';

type Props = {
  color?: string;
  size?: number;
  /** Breathing pulse (for LIVE / active states). */
  pulse?: boolean;
  glow?: boolean;
};

export function StatusDot({ color = palette.green, size = 8, pulse, glow = true }: Props) {
  const t = useSharedValue(1);

  useEffect(() => {
    if (pulse) {
      t.value = withRepeat(
        withTiming(0.3, { duration: 900, easing: Easing.inOut(Easing.quad) }),
        -1,
        true,
      );
    } else {
      t.value = 1;
    }
  }, [pulse, t]);

  const style = useAnimatedStyle(() => ({ opacity: t.value }));

  return (
    <Animated.View
      style={[
        styles.dot,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: color,
          shadowColor: color,
          shadowOpacity: glow ? 0.9 : 0,
          shadowRadius: glow ? size : 0,
        },
        style,
      ]}
    />
  );
}

const styles = StyleSheet.create({
  dot: {
    shadowOffset: { width: 0, height: 0 },
  },
});
