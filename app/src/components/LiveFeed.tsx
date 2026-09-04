/**
 * The live video surface. Renders the latest JPEG frame pushed over the
 * WebSocket (useLiveSocket writes it to the live store) with a pulsing LIVE
 * badge, a bottom scrim for legibility, and graceful connecting/offline states.
 */
import React from 'react';
import { ActivityIndicator, Image, Text, View, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn } from 'react-native-reanimated';
import { palette, radii, spacing, typography, gradients, shadows } from '@/theme';
import { useLive } from '@/stores/live';
import { StatusDot } from './StatusDot';

type Props = {
  /** Aspect ratio of the surface (16:9 default). */
  aspectRatio?: number;
  rounded?: boolean;
  /** Content overlaid on top of the feed (e.g. status chips). */
  overlay?: React.ReactNode;
};

export function LiveFeed({ aspectRatio = 16 / 9, rounded = true, overlay }: Props) {
  const frameUri = useLive((s) => s.frameUri);
  const connection = useLive((s) => s.connection);
  const isOpen = connection === 'open';

  return (
    <View style={[styles.wrap, { aspectRatio }, rounded && styles.rounded]}>
      {frameUri ? (
        <Animated.Image
          entering={FadeIn.duration(220)}
          source={{ uri: frameUri }}
          style={styles.frame}
          resizeMode="contain"
        />
      ) : (
        <View style={styles.placeholder}>
          {connection === 'closed' ? (
            <>
              <Ionicons name="cloud-offline" size={30} color={palette.textTertiary} />
              <Text style={styles.placeholderText}>Feed offline</Text>
            </>
          ) : (
            <>
              <ActivityIndicator color={palette.accent} />
              <Text style={styles.placeholderText}>Acquiring live feed…</Text>
            </>
          )}
        </View>
      )}

      {/* Bottom scrim keeps any overlaid text legible over bright frames. */}
      <LinearGradient colors={gradients.scrim} style={styles.scrim} pointerEvents="none" />

      {/* LIVE badge */}
      <View style={[styles.live, !isOpen && styles.liveDim]}>
        <StatusDot color={isOpen ? palette.red : palette.textTertiary} pulse={isOpen} size={7} />
        <Text style={[styles.liveText, !isOpen && { color: palette.textTertiary }]}>LIVE</Text>
      </View>

      {overlay ? <View style={styles.overlay}>{overlay}</View> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: '100%',
    backgroundColor: palette.black,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    overflow: 'hidden',
    justifyContent: 'center',
    // The system's single drop-shadow, spent here and nowhere else: the live
    // feed is this app's "product resting on a surface".
    ...shadows.product,
  },
  rounded: { borderRadius: radii.lg },
  frame: { width: '100%', height: '100%' },
  placeholder: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  placeholderText: { ...typography.caption, color: palette.textTertiary },
  scrim: { position: 'absolute', left: 0, right: 0, bottom: 0, height: '40%' },
  live: {
    position: 'absolute',
    top: spacing.md,
    left: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: 5,
    borderRadius: radii.sm,
    backgroundColor: palette.redDim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.redBorder,
  },
  liveDim: {
    backgroundColor: palette.chipTranslucent,
    borderColor: palette.hairlineStrong,
  },
  liveText: { ...typography.micro, color: palette.red },
  overlay: {
    position: 'absolute',
    right: spacing.md,
    bottom: spacing.md,
    left: spacing.md,
  },
});
