/**
 * One incident in a list: thumbnail, headline, AI description, severity/action
 * badges, and a "who" strip when a face was matched or captured. Entrance is a
 * staggered fade+rise (index-driven) so a fresh list settles in nicely.
 */
import React from 'react';
import { Image, Text, View, StyleSheet } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { palette, radii, spacing, typography, anomalyInfo, motion } from '@/theme';
import type { Incident } from '@/api/types';
import { evidenceUrl } from '@/api/evidence';
import { timeOfDay, percent } from '@/utils/format';
import { Card } from './Card';
import { SeverityBadge, ActionBadge } from './Badge';

type Props = {
  incident: Incident;
  index?: number;
  onPress?: () => void;
};

export function IncidentCard({ incident, index = 0, onPress }: Props) {
  const info = anomalyInfo(incident.anomaly_type);
  const fm = incident.face_match;
  const thumb = evidenceUrl(incident.thumb);
  const accent = palette[info.color] ?? palette.textSecondary;

  return (
    <Animated.View entering={FadeInDown.delay(Math.min(index, 8) * motion.stagger).springify().damping(18)}>
      <Card onPress={onPress}>
        <View style={styles.top}>
          <View style={styles.thumbWrap}>
            {thumb ? (
              <Image source={{ uri: thumb }} style={styles.thumb} />
            ) : (
              <View style={[styles.thumb, styles.thumbFallback]}>
                <Ionicons name={info.icon as keyof typeof Ionicons.glyphMap} size={22} color={accent} />
              </View>
            )}
          </View>

          <View style={styles.meta}>
            <View style={styles.headlineRow}>
              <Ionicons name={info.icon as keyof typeof Ionicons.glyphMap} size={15} color={accent} />
              <Text style={styles.headline} numberOfLines={1}>
                {incident.headline || info.label}
              </Text>
              <Text style={styles.time}>{timeOfDay(incident.ts)}</Text>
            </View>

            {incident.reasoning?.description ? (
              <Text style={styles.desc} numberOfLines={2}>
                {incident.reasoning.description}
              </Text>
            ) : null}

            <View style={styles.badges}>
              <SeverityBadge severity={incident.reasoning?.severity} />
              <ActionBadge
                action={incident.action?.action}
                dispatched={incident.action?.dispatched}
              />
            </View>
          </View>
        </View>

        {fm?.matched ? (
          <WhoStrip
            icon="checkmark-circle"
            color={palette.green}
            imageUri={evidenceUrl(fm.face_image)}
            label={fm.name ?? 'Known'}
            note={`match ${percent(fm.similarity)}`}
          />
        ) : fm?.attempted ? (
          <WhoStrip
            icon="alert-circle"
            color={palette.amber}
            imageUri={evidenceUrl(fm.face_image)}
            label="Unidentified"
            note="face captured"
          />
        ) : null}
      </Card>
    </Animated.View>
  );
}

function WhoStrip({
  icon,
  color,
  imageUri,
  label,
  note,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
  imageUri?: string;
  label: string;
  note: string;
}) {
  return (
    <View style={styles.who}>
      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.face} />
      ) : (
        <View style={[styles.face, styles.faceFallback]}>
          <Ionicons name="person" size={16} color={palette.textTertiary} />
        </View>
      )}
      <Ionicons name={icon} size={14} color={color} />
      <Text style={[styles.whoLabel, { color }]} numberOfLines={1}>
        {label}
      </Text>
      <Text style={styles.whoNote}>{note}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  top: { flexDirection: 'row', gap: spacing.md, padding: spacing.md },
  thumbWrap: { borderRadius: radii.md, overflow: 'hidden' },
  thumb: { width: 104, height: 74, borderRadius: radii.md, backgroundColor: palette.black },
  thumbFallback: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
  },
  meta: { flex: 1, minWidth: 0, justifyContent: 'center' },
  headlineRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  headline: { ...typography.bodyStrong, color: palette.text, flex: 1 },
  time: { ...typography.caption, color: palette.textTertiary },
  desc: { ...typography.caption, color: palette.textSecondary, marginTop: 4, lineHeight: 18 },
  badges: { flexDirection: 'row', gap: spacing.xs, marginTop: spacing.sm, flexWrap: 'wrap' },
  who: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: palette.hairline,
    backgroundColor: palette.surfaceElevated,
  },
  face: { width: 30, height: 30, borderRadius: radii.pill, backgroundColor: palette.black },
  faceFallback: { alignItems: 'center', justifyContent: 'center' },
  whoLabel: { ...typography.caption, fontWeight: '700', flexShrink: 1 },
  whoNote: { ...typography.caption, color: palette.textTertiary, marginLeft: 'auto' },
});
