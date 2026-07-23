/**
 * A pressable that springs down slightly on press. The base interaction of the
 * whole app — used by cards, buttons, list rows — so every touch feels
 * physical and consistent. Built on Reanimated for 60fps on the UI thread.
 */
import React from 'react';
import { Pressable, type PressableProps, type ViewStyle, type StyleProp } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import { motion } from '@/theme';

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

type Props = PressableProps & {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  /** How far to scale down on press (default from motion tokens). */
  scaleTo?: number;
  /** Dim opacity slightly on press in addition to scaling. */
  dim?: boolean;
};

export function PressableScale({ children, style, scaleTo, dim = true, ...rest }: Props) {
  const pressed = useSharedValue(0);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      {
        scale: withSpring(
          pressed.value ? (scaleTo ?? motion.pressScale) : 1,
          motion.spring,
        ),
      },
    ],
    opacity: dim ? withSpring(pressed.value ? 0.85 : 1, motion.spring) : 1,
  }));

  return (
    <AnimatedPressable
      onPressIn={(e) => {
        pressed.value = 1;
        rest.onPressIn?.(e);
      }}
      onPressOut={(e) => {
        pressed.value = 0;
        rest.onPressOut?.(e);
      }}
      style={[animatedStyle, style]}
      {...rest}
    >
      {children}
    </AnimatedPressable>
  );
}
