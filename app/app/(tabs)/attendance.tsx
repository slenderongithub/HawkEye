/**
 * Attendance — daily roster of enrolled people recognized on the live feed.
 *
 * A date stepper drives a single "YYYY-MM-DD" selection (never past today). That
 * date feeds useAttendance(date), whose records render as a calm green roster:
 * a hero "present" count up top, then one animated Card per person. All three
 * states — loading (skeleton rows), empty (context-aware copy), and populated —
 * are designed so the screen looks intentional on first paint.
 *
 * Timestamps from the backend are epoch seconds (see @/utils/format).
 */
import React, { useCallback, useMemo, useState } from 'react';
import {
  FlatList,
  RefreshControl,
  StyleSheet,
  Text,
  View,
  type ListRenderItem,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

import {
  palette,
  spacing,
  radii,
  typography,
  motion,
  gradients,
  shadows,
} from '@/theme';
import { Screen } from '@/components/Screen';
import { AppHeader } from '@/components/AppHeader';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { Card } from '@/components/Card';
import { Badge } from '@/components/Badge';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/Skeleton';
import { PressableScale } from '@/components/PressableScale';
import { SectionHeader } from '@/components/SectionHeader';
import { useAttendance, useStatus } from '@/api/queries';
import { timeOfDay, isoDate } from '@/utils/format';
import type { AttendanceRecord } from '@/api/types';

const TAB_BAR_SPACE = 108;
const SKELETON_ROWS = 5;

/* -------------------------------------------------------------------------- */
/* Date helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * Parse a local "YYYY-MM-DD" into a Date at *local* midnight. Building the Date
 * from components (not `new Date(iso)`, which parses as UTC) keeps the calendar
 * day stable regardless of timezone.
 */
function parseIsoDate(iso: string): Date {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d);
}

/** Shift an ISO date string by whole days, returning a new ISO string. */
function shiftDate(iso: string, days: number): string {
  const d = parseIsoDate(iso);
  d.setDate(d.getDate() + days);
  return isoDate(d);
}

/** Friendly relative day: "Today" / "Yesterday" / "Wed, Jul 22". */
function relativeDayLabel(iso: string, today: string): string {
  if (iso === today) return 'Today';
  if (iso === shiftDate(today, -1)) return 'Yesterday';
  return parseIsoDate(iso).toLocaleDateString([], {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
}

/** Long-form date for the header + hero context, e.g. "Wednesday, July 22". */
function fullDateLabel(iso: string): string {
  return parseIsoDate(iso).toLocaleDateString([], {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });
}

/* -------------------------------------------------------------------------- */
/* Screen                                                                     */
/* -------------------------------------------------------------------------- */

export default function AttendanceScreen() {
  const [selectedDate, setSelectedDate] = useState<string>(() => isoDate());

  // `today` gates forward navigation; string compare works because the
  // fixed-width ISO format sorts chronologically.
  const today = isoDate();
  const isToday = selectedDate === today;
  const canGoForward = selectedDate < today;

  const { data, isLoading, refetch, isRefetching } = useAttendance(selectedDate);
  const { data: status } = useStatus();

  const records = data ?? [];
  const count = records.length;
  const enrolled = status?.enrolled ?? null;
  // Only treat this as the first-load skeleton when we truly have no data yet
  // (a refetch keeps the prior list visible, driven by the RefreshControl).
  const loadingInitial = isLoading && !data;

  const relativeLabel = useMemo(
    () => relativeDayLabel(selectedDate, today),
    [selectedDate, today],
  );
  const fullDate = useMemo(() => fullDateLabel(selectedDate), [selectedDate]);

  const goPrev = useCallback(() => setSelectedDate((d) => shiftDate(d, -1)), []);
  const goNext = useCallback(
    () => setSelectedDate((d) => (d < isoDate() ? shiftDate(d, 1) : d)),
    [],
  );
  const goToday = useCallback(() => setSelectedDate(isoDate()), []);

  // Hero context line adapts to the selected day (and enrolled roster size).
  const heroCaption = isToday
    ? enrolled != null
      ? `of ${enrolled} enrolled · seen today`
      : 'recognized on the live feed today'
    : `seen ${relativeLabel === 'Yesterday' ? 'yesterday' : `on ${relativeLabel}`}`;

  const renderItem: ListRenderItem<AttendanceRecord> = useCallback(
    ({ item, index }) => <AttendanceRow record={item} index={index} />,
    [],
  );

  const keyExtractor = useCallback(
    (item: AttendanceRecord) => `${item.person_id}:${item.date}`,
    [],
  );

  return (
    <Screen>
      <AppHeader title="Attendance" subtitle={fullDate} />
      <ConnectionBanner />

      <FlatList
        data={records}
        renderItem={renderItem}
        keyExtractor={keyExtractor}
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
          <View style={styles.header}>
            {/* Date stepper — chevrons never advance past today. */}
            <Animated.View entering={FadeIn.duration(motion.duration.base)}>
              <View style={styles.stepper}>
                <StepperButton icon="chevron-back" onPress={goPrev} />
                <PressableScale
                  onPress={goToday}
                  disabled={isToday}
                  dim={!isToday}
                  style={styles.stepperCenter}
                >
                  <Text style={styles.stepperLabel}>{relativeLabel}</Text>
                  <Text style={styles.stepperDate}>{fullDate}</Text>
                  {!isToday ? (
                    <Text style={styles.stepperReset}>Jump to today</Text>
                  ) : null}
                </PressableScale>
                <StepperButton
                  icon="chevron-forward"
                  onPress={goNext}
                  disabled={!canGoForward}
                />
              </View>
            </Animated.View>

            {/* Hero — the present count, front and center. */}
            <Hero
              count={count}
              caption={heroCaption}
              loading={loadingInitial}
            />

            {count > 0 ? (
              <SectionHeader title="Present" count={count} />
            ) : null}
          </View>
        }
        ListEmptyComponent={
          loadingInitial ? (
            <View style={styles.skeletonList}>
              {Array.from({ length: SKELETON_ROWS }).map((_, i) => (
                <AttendanceSkeletonRow key={i} />
              ))}
            </View>
          ) : isToday ? (
            <EmptyState
              icon="people-outline"
              title="No one present yet"
              subtitle="As enrolled people are recognized on the live feed, they'll check in here automatically."
            />
          ) : (
            <EmptyState
              icon="calendar-outline"
              title="No attendance recorded"
              subtitle={`No enrolled people were seen on ${fullDate}.`}
            />
          )
        }
      />
    </Screen>
  );
}

/* -------------------------------------------------------------------------- */
/* Hero card                                                                  */
/* -------------------------------------------------------------------------- */

/** Large present-count with a subtle accent glow wash + roster icon. */
function Hero({
  count,
  caption,
  loading,
}: {
  count: number;
  caption: string;
  loading: boolean;
}) {
  return (
    <Animated.View
      entering={FadeInDown.duration(motion.duration.base).springify().damping(18)}
    >
      <View style={styles.hero}>
        {/* Accent glow behind the number (gradients.surfaceGlow). */}
        <LinearGradient
          colors={gradients.surfaceGlow}
          start={{ x: 0, y: 0 }}
          end={{ x: 0.9, y: 1 }}
          style={StyleSheet.absoluteFill}
          pointerEvents="none"
        />
        <View style={styles.heroText}>
          {loading ? (
            <Skeleton width={72} height={52} radius={radii.md} />
          ) : (
            <Text style={styles.heroCount} numberOfLines={1}>
              {count}
            </Text>
          )}
          <Text style={styles.heroLabel}>present</Text>
          <Text style={styles.heroCaption} numberOfLines={1}>
            {caption}
          </Text>
        </View>
        <View style={styles.heroIcon}>
          <Ionicons name="people" size={26} color={palette.accent} />
        </View>
      </View>
    </Animated.View>
  );
}

/* -------------------------------------------------------------------------- */
/* Rows                                                                       */
/* -------------------------------------------------------------------------- */

/** One present person: check avatar + name + timing meta + "present" badge. */
function AttendanceRow({
  record,
  index,
}: {
  record: AttendanceRecord;
  index: number;
}) {
  // Stagger entrances but cap the delay so long lists don't drip in slowly.
  const delay = Math.min(index, 8) * motion.stagger;
  const sightingLabel = `${record.sightings} sighting${record.sightings === 1 ? '' : 's'}`;

  return (
    <Animated.View entering={FadeInDown.delay(delay).springify().damping(18)}>
      <Card style={styles.row}>
        <View style={styles.check}>
          <Ionicons name="checkmark" size={18} color={palette.green} />
        </View>
        <View style={styles.rowBody}>
          <Text style={styles.name} numberOfLines={1}>
            {record.name}
          </Text>
          <Text style={styles.meta} numberOfLines={1}>
            first seen {timeOfDay(record.first_seen)} · last{' '}
            {timeOfDay(record.last_seen)} · {sightingLabel}
          </Text>
        </View>
        <Badge label="present" fg={palette.green} bg={palette.greenDim} />
      </Card>
    </Animated.View>
  );
}

/** Loading placeholder that mirrors the populated row layout. */
function AttendanceSkeletonRow() {
  return (
    <View style={styles.row}>
      <Skeleton width={36} height={36} radius={radii.pill} />
      <View style={styles.rowBody}>
        <Skeleton width="45%" height={15} />
        <Skeleton width="80%" height={12} />
      </View>
      <Skeleton width={62} height={20} radius={radii.sm} />
    </View>
  );
}

/* -------------------------------------------------------------------------- */
/* Stepper button                                                             */
/* -------------------------------------------------------------------------- */

function StepperButton({
  icon,
  onPress,
  disabled,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  disabled?: boolean;
}) {
  return (
    <PressableScale
      onPress={onPress}
      disabled={disabled}
      dim={!disabled}
      style={[styles.stepperBtn, disabled && styles.stepperBtnDisabled]}
    >
      <Ionicons
        name={icon}
        size={20}
        color={disabled ? palette.textTertiary : palette.text}
      />
    </PressableScale>
  );
}

/* -------------------------------------------------------------------------- */
/* Styles                                                                     */
/* -------------------------------------------------------------------------- */

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: spacing.lg,
    paddingBottom: TAB_BAR_SPACE,
    gap: spacing.md,
  },
  header: { gap: spacing.lg, paddingTop: spacing.xs },

  // Date stepper
  stepper: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.md,
    backgroundColor: palette.surface,
    borderRadius: radii.lg,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.sm,
  },
  stepperBtn: {
    width: 40,
    height: 40,
    borderRadius: radii.md,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  stepperBtnDisabled: { opacity: 0.45 },
  stepperCenter: { flex: 1, alignItems: 'center', gap: 1 },
  stepperLabel: { ...typography.bodyStrong, color: palette.text },
  stepperDate: { ...typography.caption, color: palette.textSecondary },
  stepperReset: {
    ...typography.micro,
    color: palette.accent,
    marginTop: 2,
    letterSpacing: 0.3,
  },

  // Hero
  hero: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: palette.surfaceElevated,
    borderRadius: radii.xl,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.accentBorder,
    paddingVertical: spacing.xl,
    paddingHorizontal: spacing.xl,
    overflow: 'hidden',
    ...shadows.card,
  },
  heroText: { flex: 1 },
  heroCount: {
    ...typography.hero,
    color: palette.text,
  },
  heroLabel: {
    ...typography.label,
    color: palette.green,
    marginTop: 2,
  },
  heroCaption: {
    ...typography.caption,
    color: palette.textSecondary,
    marginTop: spacing.sm,
  },
  heroIcon: {
    width: 56,
    height: 56,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.accentDim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.accentBorder,
  },

  // Rows
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  check: {
    width: 36,
    height: 36,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.greenDim,
  },
  rowBody: { flex: 1, gap: 3 },
  name: { ...typography.bodyStrong, color: palette.text },
  meta: { ...typography.caption, color: palette.textSecondary },

  skeletonList: { gap: spacing.md },
});
