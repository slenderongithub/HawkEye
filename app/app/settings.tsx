/**
 * Settings — a root-stack modal (presented as a bottom sheet by the root Stack).
 *
 * Three concerns, one per Card:
 *   1. Server      → edit + persist the LAN base URL (useSettings). Saving
 *                    triggers an automatic socket reconnect (the live socket
 *                    keys off serverBase), so we just confirm with a toast.
 *   2. Connection  → live WebSocket phase (useLive) + backend self-report
 *                    (useHealth) rendered as label/value diagnostic rows.
 *   3. About       → wordmark, version, and a forward-looking roadmap note.
 *
 * This is NOT a tab screen: no ConnectionBanner, no tab-bar padding. It owns a
 * header row with a close button that dismisses the modal.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  Keyboard,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

import { palette, radii, spacing, typography } from '@/theme';
import { Screen } from '@/components/Screen';
import { Card } from '@/components/Card';
import { Button } from '@/components/Button';
import { StatusDot } from '@/components/StatusDot';
import { Skeleton } from '@/components/Skeleton';
import { PressableScale } from '@/components/PressableScale';
import { useSettings, DEFAULT_BASE, normalizeServerBase } from '@/stores/settings';
import { useLive, type ConnectionPhase } from '@/stores/live';
import { useHealth } from '@/api/queries';
import { toast } from '@/stores/toast';

/** WebSocket phase → dot color, label, and a one-line description. */
const CONNECTION_META: Record<
  ConnectionPhase,
  { label: string; color: string; pulse: boolean; caption: string }
> = {
  open: { label: 'Live', color: palette.green, pulse: true, caption: 'Streaming frames in real time.' },
  connecting: { label: 'Connecting', color: palette.amber, pulse: true, caption: 'Reaching the server…' },
  closed: { label: 'Offline', color: palette.red, pulse: false, caption: 'No live connection to the server.' },
  idle: { label: 'Idle', color: palette.textTertiary, pulse: false, caption: 'Socket has not started yet.' },
};

/** Friendly labels for known health components; unknown keys fall back to a
 *  generic underscore→Title-Case formatter so schema drift still reads cleanly. */
const HEALTH_LABELS: Record<string, string> = {
  running: 'Pipeline',
  stage_a_loaded: 'Stage A model',
  evidence_writable: 'Evidence store',
  db_writable: 'Database',
  identity: 'Identity',
  reasoner: 'Reasoner',
};

function healthLabel(key: string): string {
  return (
    HEALTH_LABELS[key] ??
    key
      .split('_')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ')
  );
}

export default function SettingsScreen() {
  const router = useRouter();

  // --- Server field ------------------------------------------------------
  const serverBase = useSettings((s) => s.serverBase);
  const setServerBase = useSettings((s) => s.setServerBase);
  const [value, setValue] = useState(serverBase);
  const [focused, setFocused] = useState(false);
  const [saving, setSaving] = useState(false);
  // Once the operator edits, stop mirroring the store (which may still be
  // hydrating from secure-store) so we don't clobber their in-progress typing.
  const editedRef = useRef(false);

  useEffect(() => {
    if (!editedRef.current) setValue(serverBase);
  }, [serverBase]);

  // "Dirty" = a non-empty value that differs from what's persisted.
  const dirty = value.trim().length > 0 && normalizeServerBase(value) !== serverBase;

  async function onSave() {
    const next = value.trim();
    if (!next || saving) return;
    setSaving(true);
    try {
      await setServerBase(next); // store normalizes + persists; socket reconnects
      editedRef.current = false; // resume mirroring so field shows the normalized value
      Keyboard.dismiss();
      toast.show('Server updated — reconnecting', 'success', 'sync');
    } finally {
      setSaving(false);
    }
  }

  // --- Connection + health ----------------------------------------------
  const connection = useLive((s) => s.connection);
  const conn = CONNECTION_META[connection];
  const { data: health, isLoading: healthLoading } = useHealth();
  const components = health ? Object.entries(health.components) : [];

  function dismiss() {
    if (router.canGoBack()) router.back();
    else router.replace('/(tabs)/live');
  }

  return (
    <Screen edges={{ top: true, bottom: true }}>
      {/* Header: title + close affordance (circular pressable) */}
      <Animated.View entering={FadeIn.duration(220)} style={styles.header}>
        <Text style={styles.headerTitle}>Settings</Text>
        <PressableScale onPress={dismiss} style={styles.closeBtn} scaleTo={0.9}>
          <Ionicons name="close" size={20} color={palette.textSecondary} />
        </PressableScale>
      </Animated.View>

      <ScrollView
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        contentContainerStyle={styles.content}
      >
        {/* ---------------------------------------------------------------- Server */}
        <SettingsCard icon="server" title="Server" delay={40}>
          <TextInput
            value={value}
            onChangeText={(t) => {
              editedRef.current = true;
              setValue(t);
            }}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={DEFAULT_BASE}
            placeholderTextColor={palette.textTertiary}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            returnKeyType="done"
            onSubmitEditing={onSave}
            selectionColor={palette.accent}
            style={[styles.input, focused && styles.inputFocused]}
          />
          <Text style={styles.caption}>
            Your HawkEye server's LAN address — e.g. http://192.168.1.42:8000
          </Text>
          <View style={styles.saveRow}>
            <Button
              label="Save & reconnect"
              icon="save"
              onPress={onSave}
              busy={saving}
              disabled={!dirty}
              fullWidth
            />
          </View>
        </SettingsCard>

        {/* ------------------------------------------------------------ Connection */}
        <SettingsCard icon="pulse" title="Connection" delay={90}>
          {/* Live socket phase */}
          <View style={styles.connRow}>
            <StatusDot color={conn.color} size={9} pulse={conn.pulse} glow={conn.pulse} />
            <View style={styles.connText}>
              <Text style={[styles.connLabel, { color: conn.color }]}>{conn.label}</Text>
              <Text style={styles.caption}>{conn.caption}</Text>
            </View>
          </View>

          <View style={styles.divider} />

          {/* Backend self-report (GET /api/health) */}
          {healthLoading ? (
            <View style={styles.healthList}>
              <Skeleton height={16} radius={radii.sm} />
              <Skeleton height={16} radius={radii.sm} style={{ width: '80%' }} />
              <Skeleton height={16} radius={radii.sm} style={{ width: '65%' }} />
            </View>
          ) : components.length > 0 ? (
            <View style={styles.healthList}>
              {components.map(([key, val]) => (
                <HealthRow key={key} label={healthLabel(key)} value={val} />
              ))}
            </View>
          ) : (
            <Text style={styles.healthEmpty}>Health unavailable</Text>
          )}
        </SettingsCard>

        {/* ----------------------------------------------------------------- About */}
        <Animated.View entering={FadeInDown.delay(140).springify().damping(18)}>
          <Card padded>
            <Text style={styles.wordmark}>
              Hawk<Text style={styles.wordmarkAccent}>Eye</Text>
            </Text>
            <Text style={styles.version}>Version 0.1.0 · Phase 1</Text>
            <Text style={styles.aboutLine}>
              Indoor autonomous surveillance — live monitoring, incidents, attendance & identity.
            </Text>
            <Text style={styles.aboutNote}>
              Authentication, push alerts, and drone control arrive in later phases.
            </Text>
          </Card>
        </Animated.View>
      </ScrollView>
    </Screen>
  );
}

/** A titled, animated Card used for the Server and Connection sections. */
function SettingsCard({
  icon,
  title,
  delay,
  children,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  delay: number;
  children: React.ReactNode;
}) {
  return (
    <Animated.View entering={FadeInDown.delay(delay).springify().damping(18)}>
      <Card padded>
        <View style={styles.cardHead}>
          <Ionicons name={icon} size={14} color={palette.textSecondary} />
          <Text style={styles.cardTitle}>{title}</Text>
        </View>
        {children}
      </Card>
    </Animated.View>
  );
}

/** One diagnostic row: booleans render as a green/red dot, strings as text. */
function HealthRow({ label, value }: { label: string; value: unknown }) {
  const isBool = typeof value === 'boolean';
  const ok = value === true;
  return (
    <View style={styles.healthRow}>
      <Text style={styles.healthLabel} numberOfLines={1}>
        {label}
      </Text>
      {isBool ? (
        <View style={styles.healthValue}>
          <Text style={[styles.healthStatus, { color: ok ? palette.green : palette.red }]}>
            {ok ? 'OK' : 'Down'}
          </Text>
          <StatusDot color={ok ? palette.green : palette.red} size={8} glow={false} />
        </View>
      ) : (
        <Text style={styles.healthText} numberOfLines={1}>
          {String(value ?? '—')}
        </Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
  },
  headerTitle: { ...typography.title, color: palette.text },
  closeBtn: {
    width: 38,
    height: 38,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },

  content: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl,
    gap: spacing.lg,
  },

  // Card header (icon + label)
  cardHead: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  cardTitle: { ...typography.label, color: palette.textSecondary },

  // Server field
  input: {
    ...typography.body,
    color: palette.text,
    backgroundColor: palette.surfaceElevated,
    borderRadius: radii.md,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
  },
  inputFocused: { borderColor: palette.accentBorder },
  caption: {
    ...typography.caption,
    color: palette.textTertiary,
    marginTop: spacing.sm,
  },
  saveRow: { marginTop: spacing.lg },

  // Connection phase
  connRow: { flexDirection: 'row', alignItems: 'flex-start', gap: spacing.md },
  connText: { flex: 1, gap: 2 },
  connLabel: { ...typography.bodyStrong },

  divider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: palette.hairline,
    marginVertical: spacing.md,
  },

  // Health rows
  healthList: { gap: spacing.md },
  healthRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.md,
  },
  healthLabel: { ...typography.body, color: palette.textSecondary, flexShrink: 1 },
  healthValue: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  healthStatus: { ...typography.micro },
  healthText: { ...typography.bodyStrong, color: palette.text, maxWidth: '60%' },
  healthEmpty: { ...typography.body, color: palette.textTertiary },

  // About
  wordmark: { ...typography.title, color: palette.text },
  wordmarkAccent: { color: palette.accent },
  version: {
    ...typography.caption,
    color: palette.textSecondary,
    marginTop: spacing.xs,
  },
  aboutLine: {
    ...typography.body,
    color: palette.textSecondary,
    marginTop: spacing.md,
  },
  aboutNote: {
    ...typography.caption,
    color: palette.textTertiary,
    marginTop: spacing.md,
  },
});
