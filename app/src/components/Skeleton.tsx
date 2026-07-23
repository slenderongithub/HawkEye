/** Shimmering placeholder block for loading states. */
import React, { useEffect } from 'react';
import { StyleSheet, View, type DimensionValue } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
  Easing,
} from 'react-native-reanimated';
import { palette, radii } from '@/theme';

type Props = {
  width?: DimensionValue;
  height?: number;
  radius?: number;
  style?: object;
};

export function Skeleton({ width = '100%', height = 16, radius = radii.sm, style }: Props) {
  const t = useSharedValue(0.4);

  useEffect(() => {
    t.value = withRepeat(
      withTiming(0.85, { duration: 900, easing: Easing.inOut(Easing.ease) }),
      -1,
      true,
    );
  }, [t]);

  const animated = useAnimatedStyle(() => ({ opacity: t.value }));

  return (
    <Animated.View
      style={[
        { width, height, borderRadius: radius, backgroundColor: palette.surfaceElevated },
        animated,
        style,
      ]}
    />
  );
}

/** A ready-made card-shaped skeleton for list loading. */
export function SkeletonCard() {
  return (
    <View style={styles.card}>
      <Skeleton width={104} height={70} radius={radii.md} />
      <View style={styles.lines}>
        <Skeleton width="70%" height={15} />
        <Skeleton width="90%" height={12} />
        <Skeleton width="40%" height={12} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    gap: 12,
    padding: 12,
    backgroundColor: palette.surface,
    borderRadius: radii.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  lines: { flex: 1, justifyContent: 'center', gap: 8 },
});
