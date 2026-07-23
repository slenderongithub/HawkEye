/**
 * Incidents — the full incident feed.
 *
 * A FlatList of the shared <IncidentCard> rows over useIncidents(50), with a
 * client-side severity filter in the list header. The three surface states are
 * all designed for a premium first render:
 *   - loading  → shimmering SkeletonCards
 *   - empty    → a calm "all clear" (or "no matches") EmptyState
 *   - populated→ staggered, spring-entered incident cards
 *
 * Filtering is purely local (by `incident.reasoning?.severity`) so switching
 * chips is instant and never re-hits the network; the ws handler keeps the
 * underlying query cache live in the background.
 */
import React, { useMemo, useState } from 'react';
import {
  FlatList,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
  type ListRenderItem,
} from 'react-native';
import { useRouter } from 'expo-router';
import Animated, { FadeIn } from 'react-native-reanimated';

import { palette, spacing, radii, typography, motion, severityColors } from '@/theme';
import { Screen } from '@/components/Screen';
import { AppHeader } from '@/components/AppHeader';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { SectionHeader } from '@/components/SectionHeader';
import { IncidentCard } from '@/components/IncidentCard';
import { EmptyState } from '@/components/EmptyState';
import { SkeletonCard } from '@/components/Skeleton';
import { PressableScale } from '@/components/PressableScale';
import { Button } from '@/components/Button';
import { useIncidents } from '@/api/queries';
import type { Incident, Severity } from '@/api/types';

/** Padding that clears the floating tab bar so the last card isn't obscured. */
const TAB_BAR_SPACE = 108;

type FilterKey = 'all' | Severity;

/** Severity filter chips, in escalating order. Order drives the header row. */
const FILTERS: { key: FilterKey; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'critical', label: 'Critical' },
  { key: 'high', label: 'High' },
  { key: 'medium', label: 'Medium' },
  { key: 'low', label: 'Low' },
];

/** Meaningful dot color per filter — accent for "All", semantic for severities. */
function dotColorFor(key: FilterKey): string {
  if (key === 'all') return palette.accent;
  const c = severityColors(key);
  // `solid` severities (critical) carry their color on `bg`; others on `fg`.
  return c.solid ? c.bg : c.fg;
}

export default function IncidentsScreen() {
  const router = useRouter();
  const { data, isLoading, refetch, isRefetching } = useIncidents(50);

  // Explicit annotation keeps this file strict even though the shared query
  // hook's return type is currently widened to `any` upstream.
  const incidents: Incident[] = data ?? [];
  const [filter, setFilter] = useState<FilterKey>('all');

  // Client-side severity filter. Incidents without reasoning have no severity,
  // so they only appear under "All".
  const filtered = useMemo(
    () =>
      filter === 'all'
        ? incidents
        : incidents.filter((i) => i.reasoning?.severity === filter),
    [incidents, filter],
  );

  const activeLabel = FILTERS.find((f) => f.key === filter)?.label ?? 'All';

  const renderItem: ListRenderItem<Incident> = ({ item, index }) => (
    <IncidentCard
      incident={item}
      index={index}
      onPress={() => router.push(`/incident/${item.id}`)}
    />
  );

  return (
    <Screen>
      <AppHeader title="Incidents" />
      <ConnectionBanner />

      <FlatList
        data={filtered}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl
            refreshing={isRefetching}
            onRefresh={refetch}
            tintColor={palette.accent}
            colors={[palette.accent]}
            progressBackgroundColor={palette.surfaceElevated}
          />
        }
        ListHeaderComponent={
          <Animated.View entering={FadeIn.duration(motion.duration.base)} style={styles.header}>
            {/* Horizontal, non-wrapping severity filter; bleeds to screen edges. */}
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              style={styles.filterScroll}
              contentContainerStyle={styles.filterRow}
            >
              {FILTERS.map((f) => (
                <FilterChip
                  key={f.key}
                  label={f.label}
                  dotColor={dotColorFor(f.key)}
                  active={filter === f.key}
                  onPress={() => setFilter(f.key)}
                />
              ))}
            </ScrollView>

            {/* Count reflects the active filter; hidden until first load resolves. */}
            <SectionHeader
              title={`${activeLabel} incidents`}
              count={isLoading ? undefined : filtered.length}
            />
          </Animated.View>
        }
        ListEmptyComponent={
          isLoading ? (
            <View style={styles.skeletons}>
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
            </View>
          ) : (
            <Animated.View entering={FadeIn.duration(motion.duration.base)}>
              {incidents.length === 0 ? (
                // Nothing has ever been flagged — the reassuring "all clear".
                <EmptyState
                  icon="shield-checkmark"
                  title="No incidents"
                  subtitle="All clear — nothing has been flagged. HawkEye is monitoring the feed in real time."
                />
              ) : (
                // There are incidents, just none in this severity bucket.
                <EmptyState
                  icon="funnel-outline"
                  title={`No ${activeLabel.toLowerCase()} incidents`}
                  subtitle="Nothing matches this severity right now. Try another filter."
                >
                  <View style={styles.emptyAction}>
                    <Button
                      label="Show all incidents"
                      variant="secondary"
                      icon="layers-outline"
                      onPress={() => setFilter('all')}
                    />
                  </View>
                </EmptyState>
              )}
            </Animated.View>
          )
        }
      />
    </Screen>
  );
}

/** A pill in the severity filter row: colored dot + label, accent when active. */
function FilterChip({
  label,
  dotColor,
  active,
  onPress,
}: {
  label: string;
  dotColor: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <PressableScale
      onPress={onPress}
      style={[styles.chip, active ? styles.chipActive : styles.chipIdle]}
    >
      <View style={[styles.chipDot, { backgroundColor: dotColor }]} />
      <Text style={[styles.chipText, active ? styles.chipTextActive : styles.chipTextIdle]}>
        {label}
      </Text>
    </PressableScale>
  );
}

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: spacing.lg,
    paddingBottom: TAB_BAR_SPACE,
    gap: spacing.md,
    flexGrow: 1, // let the empty state breathe on short lists
  },
  header: { gap: spacing.md },
  // Negative margin cancels the list's horizontal padding so chips scroll edge-to-edge.
  filterScroll: { marginHorizontal: -spacing.lg },
  filterRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: spacing.md,
    paddingVertical: 7,
    borderRadius: radii.pill,
    borderWidth: StyleSheet.hairlineWidth,
  },
  chipActive: { backgroundColor: palette.accentDim, borderColor: palette.accentBorder },
  chipIdle: { backgroundColor: palette.surfaceElevated, borderColor: palette.hairline },
  chipDot: { width: 7, height: 7, borderRadius: radii.pill },
  chipText: { ...typography.caption, fontWeight: '700' },
  chipTextActive: { color: palette.accent },
  chipTextIdle: { color: palette.textSecondary },
  skeletons: { gap: spacing.md },
  emptyAction: { marginTop: spacing.sm },
});
