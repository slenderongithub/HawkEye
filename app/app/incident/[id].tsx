/**
 * Incident detail — a root-stack screen (not a tab) opened from any incident
 * list via `/incident/<id>`. It reads the full incident from the cached history
 * (there is NO GET-by-id endpoint on the backend) and lays out the complete
 * forensic picture: hero evidence frame, status badges, the AI assessment, who
 * (if anyone) was identified, the raw stage-A detection, and the dispatch
 * response. Every optional sub-object is guarded so a partial incident still
 * renders cleanly.
 */
import React from 'react';
import { Image, ScrollView, Text, View, StyleSheet } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

import {
  palette,
  radii,
  spacing,
  typography,
  gradients,
  motion,
  anomalyInfo,
  statusColors,
} from '@/theme';
import { Screen } from '@/components/Screen';
import { Card } from '@/components/Card';
import { Badge, SeverityBadge, ActionBadge } from '@/components/Badge';
import { Button } from '@/components/Button';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/Skeleton';
import { PressableScale } from '@/components/PressableScale';
import { useIncidents } from '@/api/queries';
import { evidenceUrl } from '@/api/evidence';
import { relativeTime, percent, titleCase } from '@/utils/format';
import type { Incident } from '@/api/types';

type IconName = keyof typeof Ionicons.glyphMap;

export default function IncidentDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();

  // No GET-by-id endpoint exists — pull the full incident out of the cached
  // history list. A fresh deep-link may arrive before the list is fetched, so
  // `isLoading` drives the skeleton state until the cache is populated.
  const { data, isLoading } = useIncidents();
  const incident = data?.find((i: Incident) => i.id === id);

  // --- Loading: header + a couple of skeletons while the cache warms up. ---
  if (isLoading && !incident) {
    return (
      <Screen edges={{ top: true, bottom: true }}>
        <HeaderBar onBack={() => router.back()} title={<Skeleton width={150} height={20} />} />
        <View style={styles.loadingBody}>
          {/* ~16:9 hero placeholder (fixed height ≈ full-width aspect ratio). */}
          <Skeleton height={210} radius={radii.lg} />
          <View style={styles.loadingRow}>
            <Skeleton width={90} height={22} radius={radii.sm} />
            <Skeleton width={90} height={22} radius={radii.sm} />
          </View>
          <Skeleton height={120} radius={radii.lg} />
          <Skeleton height={96} radius={radii.lg} />
        </View>
      </Screen>
    );
  }

  // --- Not found: cache resolved but no matching id (stale link / purged). ---
  if (!incident) {
    return (
      <Screen edges={{ top: true, bottom: true }}>
        <HeaderBar onBack={() => router.back()} title="Incident" />
        <View style={styles.notFound}>
          <EmptyState
            icon="help-circle"
            title="Incident not found"
            subtitle="This incident is no longer in the recent history."
          >
            <View style={styles.notFoundBtn}>
              <Button label="Go back" icon="arrow-back" variant="secondary" onPress={() => router.back()} />
            </View>
          </EmptyState>
        </View>
      </Screen>
    );
  }

  const info = anomalyInfo(incident.anomaly_type);
  const status = statusColors(incident.status);

  return (
    <Screen edges={{ top: true, bottom: true }}>
      <HeaderBar
        onBack={() => router.back()}
        title={info.label}
        right={<Text style={styles.headerTime}>{relativeTime(incident.ts)}</Text>}
      />

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.body}>
        {/* HERO — evidence frame with an anomaly icon + headline over a scrim. */}
        <Hero incident={incident} />

        {/* Badges row: severity · status · dispatch action. */}
        <Animated.View
          entering={FadeInDown.delay(1 * motion.stagger).springify().damping(18)}
          style={styles.badgeRow}
        >
          <SeverityBadge severity={incident.reasoning?.severity} />
          <Badge label={titleCase(incident.status)} fg={status.fg} bg={status.bg} />
          <ActionBadge action={incident.action?.action} dispatched={incident.action?.dispatched} />
        </Animated.View>

        {/* AI ASSESSMENT — purple accent for the reasoning layer. */}
        <SectionCard index={2} icon="sparkles" title="AI assessment" accent={palette.purple} accentBg={palette.purpleDim}>
          {incident.reasoning ? (
            <>
              {incident.reasoning.description ? (
                <Text style={styles.assessment}>{incident.reasoning.description}</Text>
              ) : null}
              <Row label="Confidence" value={percent(incident.reasoning.confidence)} />
              <Row label="Backend" value={incident.reasoning.backend || '—'} />
              <Row
                label="Confirmed"
                value={<YesNo value={incident.reasoning.confirmed} />}
              />
            </>
          ) : (
            <Text style={styles.muted}>No assessment</Text>
          )}
        </SectionCard>

        {/* IDENTITY — who was seen, if the face pipeline ran. */}
        <SectionCard index={3} icon="person" title="Identity" accent={palette.accent} accentBg={palette.accentDim}>
          <Identity incident={incident} />
        </SectionCard>

        {/* DETECTION — the raw stage-A trigger that opened this incident. */}
        <SectionCard index={4} icon="scan" title="Detection" accent={palette.cyan} accentBg={palette.cyanDim}>
          {incident.stage_a ? (
            <>
              <Row label="Score" value={percent(incident.stage_a.score)} />
              <Row label="Motion" value={percent(incident.stage_a.motion_score)} />
              <Row label="People" value={String(incident.stage_a.person_count)} />
              <Row label="Reason" value={incident.stage_a.reason || '—'} />
              <Row label="Triggered" value={<YesNo value={incident.stage_a.triggered} />} />
            </>
          ) : (
            <Text style={styles.muted}>No detection data</Text>
          )}
        </SectionCard>

        {/* RESPONSE — the dispatch decision + who was notified. */}
        <SectionCard index={5} icon="megaphone" title="Response" accent={palette.amber} accentBg={palette.amberDim}>
          {incident.action ? (
            <>
              <Row label="Action" value={titleCase(incident.action.action)} />
              <Row
                label="Dispatched"
                value={<YesNo value={incident.action.dispatched} yesColor={palette.red} />}
              />
              <Row label="Dispatchers" value={joinList(incident.action.dispatchers)} />
              <Row label="Reason" value={incident.action.reason || '—'} />
              <Row label="Severity" value={titleCase(incident.action.severity)} />
            </>
          ) : (
            <Text style={styles.muted}>No response taken</Text>
          )}
        </SectionCard>

        {/* Footer — absolute timestamp for the record. */}
        <Animated.Text entering={FadeIn.delay(6 * motion.stagger)} style={styles.footer}>
          {new Date(incident.ts * 1000).toLocaleString()}
        </Animated.Text>
      </ScrollView>
    </Screen>
  );
}

/* ------------------------------------------------------------------ */
/* Header                                                              */
/* ------------------------------------------------------------------ */

/** Top row shared by every state: round back button, title, optional right slot. */
function HeaderBar({
  onBack,
  title,
  right,
}: {
  onBack: () => void;
  title: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <View style={styles.header}>
      <PressableScale onPress={onBack} style={styles.backBtn} hitSlop={8}>
        <Ionicons name="arrow-back" size={20} color={palette.text} />
      </PressableScale>
      {typeof title === 'string' ? (
        <Text style={styles.headerTitle} numberOfLines={1}>
          {title}
        </Text>
      ) : (
        <View style={styles.headerTitleSlot}>{title}</View>
      )}
      {right ? <View style={styles.headerRight}>{right}</View> : null}
    </View>
  );
}

/* ------------------------------------------------------------------ */
/* Hero                                                               */
/* ------------------------------------------------------------------ */

/** Full-width ~16:9 evidence frame; falls back to a centered anomaly glyph. */
function Hero({ incident }: { incident: Incident }) {
  const info = anomalyInfo(incident.anomaly_type);
  const accent = palette[info.color] ?? palette.textSecondary;
  const thumb = evidenceUrl(incident.thumb);
  const headline = incident.headline || info.label;

  return (
    <Animated.View entering={FadeIn.duration(280)} style={styles.hero}>
      {thumb ? (
        <Image source={{ uri: thumb }} style={styles.heroImage} resizeMode="cover" />
      ) : (
        <View style={styles.heroFallback}>
          <Ionicons name={info.icon as IconName} size={48} color={accent} />
        </View>
      )}

      {/* Scrim keeps the overlaid headline legible over any frame. */}
      <LinearGradient colors={gradients.scrim} style={styles.heroScrim} pointerEvents="none" />

      <View style={styles.heroOverlay}>
        <Ionicons name={info.icon as IconName} size={18} color={accent} />
        <Text style={styles.heroHeadline} numberOfLines={2}>
          {headline}
        </Text>
      </View>
    </Animated.View>
  );
}

/* ------------------------------------------------------------------ */
/* Identity block                                                     */
/* ------------------------------------------------------------------ */

/** Face crop + verdict: matched (green), attempted-but-unknown (amber), none. */
function Identity({ incident }: { incident: Incident }) {
  const fm = incident.face_match;

  if (fm?.matched) {
    return (
      <FaceRow
        imageUri={evidenceUrl(fm.face_image)}
        icon="checkmark-circle"
        color={palette.green}
        label={fm.name ?? 'Known person'}
        note={`match ${percent(fm.similarity)}`}
      />
    );
  }

  if (fm?.attempted) {
    return (
      <FaceRow
        imageUri={evidenceUrl(fm.face_image)}
        icon="alert-circle"
        color={palette.amber}
        label="Unidentified"
        note="face captured, no match"
      />
    );
  }

  return <Text style={styles.muted}>No face captured</Text>;
}

/** Circular face crop (or person-icon fallback) beside a colored verdict. */
function FaceRow({
  imageUri,
  icon,
  color,
  label,
  note,
}: {
  imageUri?: string;
  icon: IconName;
  color: string;
  label: string;
  note: string;
}) {
  return (
    <View style={styles.faceRow}>
      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.face} />
      ) : (
        <View style={[styles.face, styles.faceFallback]}>
          <Ionicons name="person" size={20} color={palette.textTertiary} />
        </View>
      )}
      <View style={styles.faceMeta}>
        <View style={styles.faceLabelRow}>
          <Ionicons name={icon} size={15} color={color} />
          <Text style={[styles.faceLabel, { color }]} numberOfLines={1}>
            {label}
          </Text>
        </View>
        <Text style={styles.faceNote}>{note}</Text>
      </View>
    </View>
  );
}

/* ------------------------------------------------------------------ */
/* Section scaffold + rows                                            */
/* ------------------------------------------------------------------ */

/** A titled card with a left accent bar + tinted icon chip; body is children. */
function SectionCard({
  index,
  icon,
  title,
  accent,
  accentBg,
  children,
}: {
  index: number;
  icon: IconName;
  title: string;
  accent: string;
  accentBg: string;
  children: React.ReactNode;
}) {
  return (
    <Animated.View entering={FadeInDown.delay(index * motion.stagger).springify().damping(18)}>
      <Card>
        {/* Thin accent bar bleeds to the card's rounded edges (overflow hidden). */}
        <View style={[styles.accentBar, { backgroundColor: accent }]} />
        <View style={styles.sectionInner}>
          <View style={styles.sectionHeader}>
            <View style={[styles.sectionIcon, { backgroundColor: accentBg }]}>
              <Ionicons name={icon} size={14} color={accent} />
            </View>
            <Text style={styles.sectionTitle}>{title}</Text>
          </View>
          <View style={styles.sectionBody}>{children}</View>
        </View>
      </Card>
    </Animated.View>
  );
}

/** Reusable label/value pair. `value` is a node so callers can color it. */
function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      {typeof value === 'string' || typeof value === 'number' ? (
        <Text style={styles.rowValue} numberOfLines={2}>
          {value}
        </Text>
      ) : (
        <View style={styles.rowValueSlot}>{value}</View>
      )}
    </View>
  );
}

/** Green/red "Yes" vs muted "No" — the app's convention for boolean fields. */
function YesNo({ value, yesColor = palette.green }: { value?: boolean; yesColor?: string }) {
  return (
    <Text style={[styles.rowValue, { color: value ? yesColor : palette.textTertiary }]}>
      {value ? 'Yes' : 'No'}
    </Text>
  );
}

/** Join a dispatcher list defensively (default [] from zod, but guard anyway). */
function joinList(items: string[] | undefined): string {
  const list = items ?? [];
  return list.length ? list.join(', ') : '—';
}

const styles = StyleSheet.create({
  /* Header */
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  backBtn: {
    width: 38,
    height: 38,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  headerTitle: { ...typography.title, color: palette.text, flex: 1 },
  headerTitleSlot: { flex: 1 },
  headerRight: { marginLeft: 'auto' },
  headerTime: { ...typography.caption, color: palette.textTertiary },

  /* Body */
  body: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xs,
    paddingBottom: spacing.xxl,
    gap: spacing.md,
  },

  /* Hero */
  hero: {
    width: '100%',
    aspectRatio: 16 / 9,
    borderRadius: radii.lg,
    overflow: 'hidden',
    backgroundColor: palette.black,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
    justifyContent: 'flex-end',
  },
  heroImage: { ...StyleSheet.absoluteFillObject, width: '100%', height: '100%' },
  heroFallback: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center' },
  heroScrim: { position: 'absolute', left: 0, right: 0, bottom: 0, height: '55%' },
  heroOverlay: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    padding: spacing.md,
  },
  heroHeadline: { ...typography.bodyStrong, color: palette.white, flex: 1 },

  /* Badges */
  badgeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.xs },

  /* Section card */
  accentBar: { position: 'absolute', left: 0, top: 0, bottom: 0, width: 3 },
  sectionInner: { padding: spacing.lg, paddingLeft: spacing.lg + 3 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.md },
  sectionIcon: {
    width: 26,
    height: 26,
    borderRadius: radii.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sectionTitle: { ...typography.label, color: palette.textSecondary },
  sectionBody: { gap: spacing.sm },

  /* Assessment text */
  assessment: { ...typography.body, color: palette.text, marginBottom: spacing.xs },
  muted: { ...typography.body, color: palette.textTertiary },

  /* Rows */
  row: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: spacing.md,
  },
  rowLabel: { ...typography.caption, color: palette.textSecondary },
  rowValue: { ...typography.caption, color: palette.text, fontWeight: '700', flexShrink: 1, textAlign: 'right' },
  rowValueSlot: { flexShrink: 1, alignItems: 'flex-end' },

  /* Identity */
  faceRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  face: { width: 46, height: 46, borderRadius: radii.pill, backgroundColor: palette.black },
  faceFallback: { alignItems: 'center', justifyContent: 'center', backgroundColor: palette.surfaceElevated },
  faceMeta: { flex: 1, minWidth: 0, gap: 2 },
  faceLabelRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  faceLabel: { ...typography.bodyStrong, flexShrink: 1 },
  faceNote: { ...typography.caption, color: palette.textTertiary },

  /* Footer */
  footer: {
    ...typography.caption,
    color: palette.textTertiary,
    textAlign: 'center',
    marginTop: spacing.xs,
  },

  /* Loading + not-found */
  loadingBody: { paddingHorizontal: spacing.lg, paddingTop: spacing.xs, gap: spacing.md },
  loadingRow: { flexDirection: 'row', gap: spacing.xs },
  notFound: { flex: 1, justifyContent: 'center' },
  notFoundBtn: { marginTop: spacing.md },
});
