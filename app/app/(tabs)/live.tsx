/**
 * Live monitoring — the app's home screen. Shows the real-time video feed, a
 * strip of runtime status, and a live-updating preview of recent incidents.
 *
 * Data sources:
 *   - useLive()     → frame + connection + latest status (from the WebSocket)
 *   - useStatus()   → REST fallback/poll for status when the socket is quiet
 *   - useIncidents()→ recent incident history (kept live by the ws handler,
 *                     which prepends new incidents into this same query cache)
 */
import React from 'react';
import { ScrollView, Text, View, StyleSheet, Pressable } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn } from 'react-native-reanimated';

import { palette, spacing, typography, radii } from '@/theme';
import { Screen } from '@/components/Screen';
import { AppHeader } from '@/components/AppHeader';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { LiveFeed } from '@/components/LiveFeed';
import { SectionHeader } from '@/components/SectionHeader';
import { IncidentCard } from '@/components/IncidentCard';
import { EmptyState } from '@/components/EmptyState';
import { SkeletonCard } from '@/components/Skeleton';
import { PressableScale } from '@/components/PressableScale';
import { useLive } from '@/stores/live';
import { useStatus, useIncidents } from '@/api/queries';
import type { Incident } from '@/api/types';

const TAB_BAR_SPACE = 108;
const PREVIEW_COUNT = 4;

export default function LiveScreen() {
  const router = useRouter();
  const liveStatus = useLive((s) => s.status);
  const { data: polledStatus } = useStatus();
  const status = liveStatus ?? polledStatus;

  const { data: incidents, isLoading } = useIncidents();
  const list: Incident[] = incidents ?? [];
  const preview = list.slice(0, PREVIEW_COUNT);

  return (
    <Screen>
      <AppHeader
        subtitle="Indoor surveillance"
        right={
          <PressableScale onPress={() => router.push('/settings')} style={styles.gear}>
            <Ionicons name="settings-outline" size={18} color={palette.textSecondary} />
          </PressableScale>
        }
      />
      <ConnectionBanner />

      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.content}
      >
        <LiveFeed
          overlay={
            <View style={styles.feedOverlay}>
              <FeedPill icon="speedometer" label={fmtFps(status?.fps)} />
              <FeedPill icon="sparkles" label={status?.reasoner ?? '—'} />
            </View>
          }
        />

        {/* Runtime status strip */}
        <View style={styles.statRow}>
          <StatTile
            icon="pulse"
            label="Monitoring"
            value={status?.running ? 'Active' : 'Idle'}
            tone={status?.running ? palette.green : palette.textTertiary}
          />
          <StatTile
            icon="scan"
            label="Identity"
            value={status?.identity ? 'On' : 'Off'}
            tone={status?.identity ? palette.accent : palette.textTertiary}
          />
          <StatTile
            icon="people"
            label="Present"
            value={String(status?.present_today ?? 0)}
            tone={palette.text}
          />
        </View>

        {/* Recent activity */}
        <View style={styles.section}>
          <SectionHeader
            title="Recent activity"
            count={incidents?.length}
            right={
              <Pressable onPress={() => router.push('/(tabs)/incidents')} hitSlop={8}>
                <Text style={styles.viewAll}>View all</Text>
              </Pressable>
            }
          />

          {isLoading ? (
            <View style={styles.list}>
              <SkeletonCard />
              <SkeletonCard />
            </View>
          ) : preview.length === 0 ? (
            <EmptyState
              icon="shield-checkmark"
              title="All clear"
              subtitle="No incidents detected. The feed is being monitored in real time."
            />
          ) : (
            <View style={styles.list}>
              {preview.map((inc, i) => (
                <IncidentCard
                  key={inc.id}
                  incident={inc}
                  index={i}
                  onPress={() => router.push(`/incident/${inc.id}`)}
                />
              ))}
            </View>
          )}
        </View>
      </ScrollView>
    </Screen>
  );
}

function fmtFps(fps: number | null | undefined): string {
  if (fps == null) return '— fps';
  return `${fps.toFixed(fps < 10 ? 1 : 0)} fps`;
}

function FeedPill({ icon, label }: { icon: keyof typeof Ionicons.glyphMap; label: string }) {
  return (
    <View style={styles.feedPill}>
      <Ionicons name={icon} size={12} color={palette.text} />
      <Text style={styles.feedPillText} numberOfLines={1}>
        {label}
      </Text>
    </View>
  );
}

function StatTile({
  icon,
  label,
  value,
  tone,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  value: string;
  tone: string;
}) {
  return (
    <Animated.View entering={FadeIn.duration(260)} style={styles.tile}>
      <Ionicons name={icon} size={16} color={tone} />
      <Text style={[styles.tileValue, { color: tone }]} numberOfLines={1}>
        {value}
      </Text>
      <Text style={styles.tileLabel}>{label}</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: spacing.lg, paddingBottom: TAB_BAR_SPACE, gap: spacing.lg },
  gear: {
    width: 34,
    height: 34,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  feedOverlay: { flexDirection: 'row', gap: spacing.sm },
  feedPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: radii.sm,
    backgroundColor: palette.overlay,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
  },
  feedPillText: { ...typography.micro, color: palette.text, fontWeight: '700' },
  statRow: { flexDirection: 'row', gap: spacing.sm },
  tile: {
    flex: 1,
    backgroundColor: palette.surface,
    borderRadius: radii.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    gap: 4,
  },
  tileValue: { ...typography.heading },
  tileLabel: { ...typography.micro, color: palette.textSecondary, letterSpacing: 0.3 },
  section: { gap: spacing.xs },
  list: { gap: spacing.md },
  viewAll: { ...typography.caption, color: palette.accent, fontWeight: '700' },
});
