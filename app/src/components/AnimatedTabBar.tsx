/**
 * Custom bottom tab bar: a floating, blurred pill with an animated accent
 * indicator that springs to the active tab. Icons swap to their filled variant
 * when active. This bar is a big part of the app's perceived polish, so it's
 * hand-built rather than using the default React Navigation bar.
 */
import React from 'react';
import { Pressable, Text, View, StyleSheet, Platform } from 'react-native';
import { BlurView } from 'expo-blur';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Animated, {
  useAnimatedStyle,
  withSpring,
  interpolateColor,
  useDerivedValue,
} from 'react-native-reanimated';
import type { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { palette, radii, spacing, typography, motion } from '@/theme';

const ICONS: Record<string, { on: keyof typeof Ionicons.glyphMap; off: keyof typeof Ionicons.glyphMap; label: string }> = {
  live: { on: 'videocam', off: 'videocam-outline', label: 'Live' },
  incidents: { on: 'alert-circle', off: 'alert-circle-outline', label: 'Incidents' },
  attendance: { on: 'people', off: 'people-outline', label: 'Attendance' },
  people: { on: 'person-add', off: 'person-add-outline', label: 'People' },
  drone: { on: 'navigate', off: 'navigate-outline', label: 'Drone' },
};

function TabItem({
  routeKey,
  routeName,
  focused,
  onPress,
}: {
  routeKey: string;
  routeName: string;
  focused: boolean;
  onPress: () => void;
}) {
  const meta = ICONS[routeName] ?? { on: 'ellipse', off: 'ellipse-outline', label: routeName };
  const active = useDerivedValue(() => withSpring(focused ? 1 : 0, motion.springSoft), [focused]);

  const iconStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: withSpring(focused ? -1 : 0, motion.spring) }],
  }));
  const labelStyle = useAnimatedStyle(() => ({
    color: interpolateColor(active.value, [0, 1], [palette.textTertiary, palette.accent]),
  }));

  return (
    <Pressable style={styles.item} onPress={onPress} hitSlop={8}>
      <Animated.View style={iconStyle}>
        <Ionicons
          name={focused ? meta.on : meta.off}
          size={22}
          color={focused ? palette.accent : palette.textTertiary}
        />
      </Animated.View>
      <Animated.Text style={[styles.label, labelStyle]}>{meta.label}</Animated.Text>
    </Pressable>
  );
}

export function AnimatedTabBar({ state, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  // Only tabs we want visible in the bar (hidden routes are filtered out).
  const routes = state.routes.filter((r) => ICONS[r.name]);
  const activeName = state.routes[state.index]?.name;

  return (
    <View style={[styles.wrap, { paddingBottom: insets.bottom || spacing.md }]} pointerEvents="box-none">
      <BlurView intensity={Platform.OS === 'ios' ? 40 : 24} tint="dark" style={styles.bar}>
        {routes.map((route) => {
          const focused = route.name === activeName;
          return (
            <TabItem
              key={route.key}
              routeKey={route.key}
              routeName={route.name}
              focused={focused}
              onPress={() => {
                const event = navigation.emit({
                  type: 'tabPress',
                  target: route.key,
                  canPreventDefault: true,
                });
                if (!focused && !event.defaultPrevented) {
                  navigation.navigate(route.name);
                }
              }}
            />
          );
        })}
      </BlurView>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: spacing.lg,
    alignItems: 'center',
  },
  bar: {
    flexDirection: 'row',
    width: '100%',
    maxWidth: 520,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.sm,
    borderRadius: radii.xl,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
    backgroundColor: 'rgba(20,25,38,0.75)',
    overflow: 'hidden',
  },
  item: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 3,
    paddingVertical: 4,
  },
  label: {
    ...typography.micro,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
});
