/**
 * Drone — a contract-first scaffold for the autonomous-patrol phase that is not
 * yet wired to hardware. The goal is to look intentional and obviously "upcoming"
 * rather than broken or fake-live: no drone is paired, so telemetry shows honest
 * placeholder dashes, patrol routes are labelled Examples, and the manual D-pad
 * is rendered visually locked. Every control still gives feedback (a toast that
 * explains the roadmap) so the screen sells the plan without pretending to fly.
 */
import React from 'react';
import { ScrollView, Text, View, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown } from 'react-native-reanimated';

import { palette, spacing, typography, radii, motion } from '@/theme';
import { Screen } from '@/components/Screen';
import { AppHeader } from '@/components/AppHeader';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { Card } from '@/components/Card';
import { Badge } from '@/components/Badge';
import { Button } from '@/components/Button';
import { SectionHeader } from '@/components/SectionHeader';
import { StatusDot } from '@/components/StatusDot';
import { PressableScale } from '@/components/PressableScale';
import { toast } from '@/stores/toast';

const TAB_BAR_SPACE = 108;

type IconName = keyof typeof Ionicons.glyphMap;

/** Example patrol routes — sample content that ships before any fleet exists. */
type PatrolRoute = {
  id: string;
  name: string;
  icon: IconName;
  waypoints: number;
  minutes: number;
  color: string; // foreground accent for this route's glyph
  tint: string; // dim background for the glyph circle
};

const ROUTES: PatrolRoute[] = [
  { id: 'perimeter', name: 'Perimeter Loop', icon: 'sync', waypoints: 8, minutes: 4, color: palette.accent, tint: palette.accentDim },
  { id: 'night', name: 'Night Sweep', icon: 'moon', waypoints: 12, minutes: 7, color: palette.purple, tint: palette.purpleDim },
  { id: 'lobby', name: 'Lobby Watch', icon: 'business', waypoints: 5, minutes: 3, color: palette.cyan, tint: palette.cyanDim },
];

/** Telemetry read-out rows — real-looking layout, honest placeholder values. */
type Telemetry = {
  icon: IconName;
  label: string;
  value: string;
  valueColor: string;
  dot?: string; // optional leading StatusDot color
};

const TELEMETRY: Telemetry[] = [
  { icon: 'radio', label: 'Connection', value: 'Offline', valueColor: palette.red, dot: palette.red },
  { icon: 'battery-dead', label: 'Battery', value: '—', valueColor: palette.textTertiary },
  { icon: 'location', label: 'Position', value: '—', valueColor: palette.textTertiary },
  { icon: 'options', label: 'Mode', value: 'Standby', valueColor: palette.textSecondary },
];

/** A single spring-staggered entrance so the page assembles calmly on mount. */
function enter(index: number) {
  return FadeInDown.delay(Math.min(index, 8) * motion.stagger)
    .springify()
    .damping(18);
}

export default function DroneScreen() {
  // Manual controls are inert until a drone is paired; every tap explains why.
  const notifyLocked = () =>
    toast.show('Manual flight unlocks when a drone is paired', 'info', 'lock-closed');
  const notifyRoadmap = () =>
    toast.show('Drone control arrives in a later phase', 'info', 'navigate');

  return (
    <Screen>
      <AppHeader title="Drone" subtitle="Fleet control" />
      <ConnectionBanner />

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.content}>
        {/* Hero status — the honest "no hardware yet" empty state, badged PREVIEW. */}
        <Animated.View entering={enter(0)}>
          <Card elevated padded style={styles.hero}>
            <View style={styles.previewPill}>
              <Badge label="Preview" fg={palette.amber} bg={palette.amberDim} />
            </View>
            <View style={styles.heroBadge}>
              <Ionicons name="navigate" size={28} color={palette.accent} />
            </View>
            <Text style={styles.heroTitle}>No drone paired</Text>
            <Text style={styles.heroSubtitle}>
              Autonomous patrol & manual flight arrive in a later phase.
            </Text>
          </Card>
        </Animated.View>

        {/* Fleet status — telemetry panel with placeholder read-outs. */}
        <Animated.View entering={enter(1)}>
          <Card padded>
            <View style={styles.panelHead}>
              <Text style={styles.panelLabel}>Fleet status</Text>
              <View style={styles.telemetryTag}>
                <Ionicons name="pulse" size={11} color={palette.textTertiary} />
                <Text style={styles.telemetryTagText}>Telemetry</Text>
              </View>
            </View>
            <View style={styles.telemetryList}>
              {TELEMETRY.map((row, i) => (
                <View key={row.label} style={[styles.telemetryRow, i > 0 && styles.telemetryDivider]}>
                  <Ionicons name={row.icon} size={15} color={palette.textTertiary} style={styles.telemetryIcon} />
                  <Text style={styles.telemetryLabel}>{row.label}</Text>
                  <View style={styles.telemetryValueCell}>
                    {row.dot ? <StatusDot color={row.dot} size={7} glow={false} /> : null}
                    <Text style={[styles.telemetryValue, { color: row.valueColor }]}>{row.value}</Text>
                  </View>
                </View>
              ))}
            </View>
          </Card>
        </Animated.View>

        {/* Patrol routes — sample missions, clearly flagged as examples. */}
        <Animated.View entering={enter(2)} style={styles.section}>
          <SectionHeader
            title="Patrol routes"
            count={ROUTES.length}
            right={<Text style={styles.sectionHint}>Examples</Text>}
          />
          <View style={styles.list}>
            {ROUTES.map((route, i) => (
              <Animated.View key={route.id} entering={enter(3 + i)}>
                <Card padded style={styles.routeCard}>
                  <View style={[styles.routeGlyph, { backgroundColor: route.tint }]}>
                    <Ionicons name={route.icon} size={18} color={route.color} />
                  </View>
                  <View style={styles.routeText}>
                    <Text style={styles.routeName} numberOfLines={1}>
                      {route.name}
                    </Text>
                    <Text style={styles.routeMeta} numberOfLines={1}>
                      {route.waypoints} waypoints · ~{route.minutes} min
                    </Text>
                  </View>
                  <Button label="Start" icon="play" variant="secondary" onPress={notifyRoadmap} />
                </Card>
              </Animated.View>
            ))}
          </View>
        </Animated.View>

        {/* Manual override — a polished D-pad rendered visually locked. */}
        <Animated.View entering={enter(7)} style={styles.section}>
          <SectionHeader title="Manual override" />
          <Card padded>
            {/* Dimmed control surface; still tappable so taps can explain the lock. */}
            <View style={styles.dpadWrap}>
              <View style={styles.dpadRow}>
                <View style={styles.dpadSpacer} />
                <DPadButton icon="chevron-up" onPress={notifyLocked} />
                <View style={styles.dpadSpacer} />
              </View>
              <View style={styles.dpadRow}>
                <DPadButton icon="chevron-back" onPress={notifyLocked} />
                <DPadButton icon="locate" center onPress={notifyLocked} />
                <DPadButton icon="chevron-forward" onPress={notifyLocked} />
              </View>
              <View style={styles.dpadRow}>
                <View style={styles.dpadSpacer} />
                <DPadButton icon="chevron-down" onPress={notifyLocked} />
                <View style={styles.dpadSpacer} />
              </View>
            </View>

            {/* Crisp lock affordance sits at full opacity beneath the dimmed pad. */}
            <View style={styles.lockRow}>
              <Ionicons name="lock-closed" size={13} color={palette.amber} />
              <Text style={styles.lockText}>Locked — no drone connected</Text>
            </View>
          </Card>
        </Animated.View>
      </ScrollView>
    </Screen>
  );
}

/** One directional cell of the manual D-pad. `center` styles the hold-position key. */
function DPadButton({
  icon,
  onPress,
  center,
}: {
  icon: IconName;
  onPress: () => void;
  center?: boolean;
}) {
  return (
    <PressableScale onPress={onPress} style={[styles.dpadBtn, center && styles.dpadCenter]}>
      <Ionicons
        name={icon}
        size={center ? 20 : 22}
        color={center ? palette.accent : palette.textSecondary}
      />
    </PressableScale>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: spacing.lg, paddingBottom: TAB_BAR_SPACE, gap: spacing.lg },

  // Hero status card
  hero: { alignItems: 'center', gap: spacing.sm, paddingVertical: spacing.xl },
  previewPill: { position: 'absolute', top: spacing.md, right: spacing.md },
  heroBadge: {
    width: 64,
    height: 64,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.accentDim,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.accentBorder,
    marginBottom: spacing.xs,
  },
  heroTitle: { ...typography.heading, color: palette.text },
  heroSubtitle: {
    ...typography.caption,
    color: palette.textSecondary,
    textAlign: 'center',
    maxWidth: 280,
  },

  // Fleet status telemetry panel
  panelHead: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  panelLabel: { ...typography.label, color: palette.textSecondary },
  telemetryTag: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  telemetryTagText: { ...typography.micro, color: palette.textTertiary },
  telemetryList: {},
  telemetryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  telemetryDivider: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: palette.hairline,
  },
  telemetryIcon: { marginRight: spacing.sm, width: 18 },
  telemetryLabel: { ...typography.body, color: palette.textSecondary, flex: 1 },
  telemetryValueCell: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  telemetryValue: { ...typography.bodyStrong },

  // Sections
  section: { gap: spacing.xs },
  sectionHint: { ...typography.micro, color: palette.textTertiary },
  list: { gap: spacing.md },

  // Patrol route rows
  routeCard: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  routeGlyph: {
    width: 40,
    height: 40,
    borderRadius: radii.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  routeText: { flex: 1, gap: 2 },
  routeName: { ...typography.bodyStrong, color: palette.text },
  routeMeta: { ...typography.caption, color: palette.textSecondary },

  // Manual override D-pad
  dpadWrap: { alignItems: 'center', gap: spacing.sm, opacity: 0.45, paddingVertical: spacing.sm },
  dpadRow: { flexDirection: 'row', gap: spacing.sm },
  dpadSpacer: { width: 54, height: 54 },
  dpadBtn: {
    width: 54,
    height: 54,
    borderRadius: radii.md,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
  },
  dpadCenter: {
    borderRadius: radii.pill,
    backgroundColor: palette.accentDim,
    borderColor: palette.accentBorder,
  },
  lockRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    marginTop: spacing.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    alignSelf: 'center',
    borderRadius: radii.pill,
    backgroundColor: palette.amberDim,
  },
  lockText: { ...typography.caption, color: palette.amber, fontWeight: '700' },
});
