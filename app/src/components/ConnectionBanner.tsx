/**
 * Slide-in banner shown when the live socket is not open, so the operator knows
 * data may be stale. Animates in/out; renders nothing while connected.
 */
import React from 'react';
import { Text, StyleSheet } from 'react-native';
import Animated, { FadeInUp, FadeOutUp } from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { palette, radii, spacing, typography } from '@/theme';
import { useLive } from '@/stores/live';

export function ConnectionBanner() {
  const connection = useLive((s) => s.connection);
  if (connection === 'open' || connection === 'idle') return null;

  const connecting = connection === 'connecting';
  return (
    <Animated.View
      entering={FadeInUp.springify().damping(18)}
      exiting={FadeOutUp.duration(180)}
      style={[styles.banner, connecting ? styles.warn : styles.err]}
    >
      <Ionicons
        name={connecting ? 'sync' : 'cloud-offline'}
        size={15}
        color={connecting ? palette.amber : palette.red}
      />
      <Text style={[styles.text, { color: connecting ? palette.amber : palette.red }]}>
        {connecting ? 'Connecting to HawkEye…' : 'Offline — trying to reconnect'}
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginHorizontal: spacing.lg,
    marginBottom: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radii.md,
    borderWidth: StyleSheet.hairlineWidth,
  },
  warn: { backgroundColor: palette.amberDim, borderColor: 'rgba(255,176,32,0.35)' },
  err: { backgroundColor: palette.redDim, borderColor: 'rgba(255,77,94,0.35)' },
  text: { ...typography.caption, fontWeight: '600' },
});
