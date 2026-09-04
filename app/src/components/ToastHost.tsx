/**
 * Renders the global toast queue at the bottom of the screen. Each toast slides
 * up, lives ~3.5s, then fades out. Mounted once in the root layout.
 */
import React, { useEffect } from 'react';
import { Text, View, StyleSheet } from 'react-native';
import Animated, { FadeInDown, FadeOutDown, LinearTransition } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { palette, radii, spacing, typography } from '@/theme';
import { useToast, type Toast, type ToastTone } from '@/stores/toast';

const TONES: Record<ToastTone, { color: string; bg: string; icon: keyof typeof Ionicons.glyphMap }> = {
  success: { color: palette.green, bg: palette.greenDim, icon: 'checkmark-circle' },
  info: { color: palette.accent, bg: palette.accentDim, icon: 'information-circle' },
  alert: { color: palette.red, bg: palette.redDim, icon: 'alert-circle' },
};

function ToastRow({ toast }: { toast: Toast }) {
  const dismiss = useToast((s) => s.dismiss);
  const tone = TONES[toast.tone];

  useEffect(() => {
    const timer = setTimeout(() => dismiss(toast.id), 3500);
    return () => clearTimeout(timer);
  }, [toast.id, dismiss]);

  return (
    <Animated.View
      entering={FadeInDown.springify().damping(16)}
      exiting={FadeOutDown.duration(220)}
      layout={LinearTransition.springify()}
      style={[styles.toast, { borderColor: tone.color }]}
    >
      <Ionicons name={(toast.icon as keyof typeof Ionicons.glyphMap) ?? tone.icon} size={18} color={tone.color} />
      <Text style={styles.text} numberOfLines={2}>
        {toast.text}
      </Text>
    </Animated.View>
  );
}

export function ToastHost() {
  const toasts = useToast((s) => s.toasts);
  const insets = useSafeAreaInsets();

  return (
    <View pointerEvents="box-none" style={[styles.host, { bottom: insets.bottom + 90 }]}>
      {toasts.map((t) => (
        <ToastRow key={t.id} toast={t} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  host: {
    position: 'absolute',
    left: spacing.lg,
    right: spacing.lg,
    gap: spacing.sm,
    alignItems: 'center',
  },
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    maxWidth: 420,
    width: '100%',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: radii.md,
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
  },
  text: {
    ...typography.body,
    color: palette.text,
    flex: 1,
    fontWeight: '600',
  },
});
