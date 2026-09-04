/**
 * People — the enrolled identity roster + inline enrollment.
 *
 * Two jobs on one screen:
 *   1. Enroll: a primary CTA expands an inline form (name + optional role +
 *      one-or-more library photos) that POSTs a multipart face enrollment.
 *   2. Roster: a live FlatList of enrolled people with initials avatars, a
 *      derived meta line, and a per-row destructive delete (with confirm).
 *
 * Data: usePeople() for the roster, useEnrollPerson()/useDeletePerson() for
 * mutations. All three share the ['people', serverBase] query cache, so an
 * enroll/delete optimistically invalidates and the list re-fetches itself.
 */
import React, { useState } from 'react';
import {
  Alert,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Platform,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  type TextInputProps,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import Animated, { FadeInDown, FadeOutUp } from 'react-native-reanimated';

import { palette, radii, spacing, typography, motion } from '@/theme';
import { Screen } from '@/components/Screen';
import { AppHeader } from '@/components/AppHeader';
import { ConnectionBanner } from '@/components/ConnectionBanner';
import { Card } from '@/components/Card';
import { Button } from '@/components/Button';
import { SectionHeader } from '@/components/SectionHeader';
import { EmptyState } from '@/components/EmptyState';
import { Skeleton } from '@/components/Skeleton';
import { usePeople, useEnrollPerson, useDeletePerson } from '@/api/queries';
import { toast } from '@/stores/toast';
import type { Person } from '@/api/types';

// Clears the floating tab bar so the last row isn't hidden behind it.
const TAB_BAR_SPACE = 108;
// Cap how many selected thumbnails we lay out inline before switching to a count.
const MAX_THUMBS = 6;

export default function PeopleScreen() {
  const { data: people, isLoading, refetch, isRefetching } = usePeople();
  const enroll = useEnrollPerson();
  const del = useDeletePerson();

  // Enroll form state (lives on the screen so it survives list re-renders).
  const [expanded, setExpanded] = useState(false);
  const [name, setName] = useState('');
  const [role, setRole] = useState('');
  const [assets, setAssets] = useState<ImagePicker.ImagePickerAsset[]>([]);
  // Track the row being deleted so only its trash button spins.
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const resetForm = () => {
    setName('');
    setRole('');
    setAssets([]);
  };

  /** Ask for library permission, then let the user pick one-or-more photos. */
  const pickPhotos = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      toast.show('Photo library access is required to enroll', 'alert', 'lock-closed');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsMultipleSelection: true,
      quality: 0.7,
    });
    if (result.canceled) return;
    setAssets(result.assets ?? []);
  };

  /** Validate, submit the multipart enrollment, then reset + collapse. */
  const submit = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) {
      toast.show('Enter a name to enroll', 'alert', 'person');
      return;
    }
    if (assets.length === 0) {
      toast.show('Select at least one photo', 'alert', 'images');
      return;
    }
    const trimmedRole = role.trim();
    try {
      await enroll.mutateAsync({
        name: trimmedName,
        meta: trimmedRole ? { role: trimmedRole } : {},
        images: assets.map((a) => ({
          uri: a.uri,
          name: a.fileName ?? undefined,
          type: a.mimeType ?? 'image/jpeg',
        })),
      });
      toast.show(`${trimmedName} enrolled`, 'success', 'checkmark-circle');
      resetForm();
      setExpanded(false);
    } catch {
      toast.show('Enrollment failed. Try again.', 'alert', 'warning');
    }
  };

  /** Confirm, then remove a person and their enrolled faces. */
  const confirmDelete = (p: Person) => {
    Alert.alert(
      'Remove person',
      `Remove ${p.name} from the identity database? This deletes their enrolled faces.`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Remove', style: 'destructive', onPress: () => runDelete(p) },
      ],
    );
  };

  const runDelete = async (p: Person) => {
    setDeletingId(p.id);
    try {
      await del.mutateAsync(p.id);
      toast.show(`${p.name} removed`, 'info', 'trash');
    } catch {
      toast.show('Could not remove person', 'alert', 'warning');
    } finally {
      setDeletingId(null);
    }
  };

  // Header holds the enroll CTA, the animated form, and the roster label.
  // Passed to FlatList as a stable *element* (not a component) so the form's
  // TextInputs keep focus across the parent's per-keystroke re-renders.
  const listHeader = (
    <View style={styles.header}>
      <Button
        label="Enroll a person"
        icon="person-add"
        onPress={() => setExpanded((v) => !v)}
      />

      {expanded ? (
        <Animated.View
          entering={FadeInDown.springify().damping(18)}
          exiting={FadeOutUp.duration(motion.duration.fast)}
        >
          <Card padded style={styles.form}>
            <Text style={styles.formHint}>
              Add clear, front-facing photos. More angles improve recognition.
            </Text>

            <LabeledInput
              label="Full name"
              value={name}
              onChangeText={setName}
              placeholder="e.g. Priya Sharma"
              autoCapitalize="words"
              returnKeyType="next"
            />
            <LabeledInput
              label="Role / ID (optional)"
              value={role}
              onChangeText={setRole}
              placeholder="e.g. Security · Badge 214"
              autoCapitalize="words"
              returnKeyType="done"
            />

            <View style={styles.photoRow}>
              <View style={styles.photoBtn}>
                <Button label="Select photos" icon="images" variant="secondary" onPress={pickPhotos} />
              </View>
              {assets.length > 0 ? (
                <Text style={styles.photoCount}>
                  {assets.length} selected
                </Text>
              ) : null}
            </View>

            {/* Thumbnail strip of the chosen assets (premium confirmation). */}
            {assets.length > 0 ? (
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.thumbStrip}
              >
                {assets.slice(0, MAX_THUMBS).map((a, i) => (
                  <Image key={`${a.uri}-${i}`} source={{ uri: a.uri }} style={styles.thumb} />
                ))}
                {assets.length > MAX_THUMBS ? (
                  <View style={[styles.thumb, styles.thumbMore]}>
                    <Text style={styles.thumbMoreText}>+{assets.length - MAX_THUMBS}</Text>
                  </View>
                ) : null}
              </ScrollView>
            ) : null}

            <Button
              label="Add to identity DB"
              icon="cloud-upload"
              busy={enroll.isPending}
              fullWidth
              onPress={submit}
            />
          </Card>
        </Animated.View>
      ) : null}

      <SectionHeader title="Enrolled" count={people?.length} />
    </View>
  );

  return (
    <Screen>
      <AppHeader title="People" />
      <ConnectionBanner />

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <FlatList<Person>
          data={people ?? []}
          keyExtractor={(p) => p.id}
          ListHeaderComponent={listHeader}
          renderItem={({ item, index }) => (
            <PersonRow
              person={item}
              index={index}
              deleting={deletingId === item.id}
              onDelete={() => confirmDelete(item)}
            />
          )}
          ListEmptyComponent={
            isLoading ? (
              <View style={styles.list}>
                {[0, 1, 2].map((k) => (
                  <PersonSkeleton key={k} />
                ))}
              </View>
            ) : (
              <EmptyState
                icon="people-outline"
                title="No one enrolled yet"
                subtitle="Enroll a person to start recognizing faces in the live feed."
              />
            )
          }
          contentContainerStyle={styles.content}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={isRefetching}
              onRefresh={refetch}
              tintColor={palette.accent}
              colors={[palette.accent]}
              progressBackgroundColor={palette.surfaceElevated}
            />
          }
        />
      </KeyboardAvoidingView>
    </Screen>
  );
}

/* ----------------------------- Person row ------------------------------- */

function PersonRow({
  person,
  index,
  deleting,
  onDelete,
}: {
  person: Person;
  index: number;
  deleting: boolean;
  onDelete: () => void;
}) {
  const glyphs = initials(person.name);
  return (
    <Animated.View
      entering={FadeInDown.delay(Math.min(index, 8) * motion.stagger)
        .springify()
        .damping(18)}
    >
      <Card style={styles.personCard}>
        <View style={styles.avatar}>
          {glyphs ? (
            <Text style={styles.avatarText}>{glyphs}</Text>
          ) : (
            <Ionicons name="person" size={20} color={palette.textSecondary} />
          )}
        </View>
        <View style={styles.personBody}>
          <Text style={styles.personName} numberOfLines={1}>
            {person.name}
          </Text>
          <Text style={styles.personMeta} numberOfLines={1}>
            {metaLine(person)}
          </Text>
        </View>
        <Button
          label=""
          icon="trash-outline"
          variant="danger"
          busy={deleting}
          onPress={onDelete}
        />
      </Card>
    </Animated.View>
  );
}

/* ------------------------------ Skeleton -------------------------------- */

/** Row-shaped placeholder that mirrors the PersonRow layout while loading. */
function PersonSkeleton() {
  return (
    <View style={styles.personCard}>
      <Skeleton width={44} height={44} radius={radii.pill} />
      <View style={styles.personBody}>
        <Skeleton width="55%" height={15} />
        <Skeleton width="35%" height={12} style={{ marginTop: spacing.xs }} />
      </View>
      <Skeleton width={50} height={44} radius={radii.md} />
    </View>
  );
}

/* --------------------------- Labeled input ------------------------------ */
// Defined at module scope (stable component identity) so the TextInput isn't
// remounted — and doesn't lose focus — when the screen re-renders on keystroke.
function LabeledInput({
  label,
  ...inputProps
}: { label: string } & TextInputProps) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput
        style={styles.input}
        placeholderTextColor={palette.textTertiary}
        selectionColor={palette.accent}
        {...inputProps}
      />
    </View>
  );
}

/* ------------------------------ Helpers --------------------------------- */

/** Up-to-two-letter initials from a name (first + last word). */
function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return '';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

/** Meta line: joined meta entries, or a face count as the fallback. */
function metaLine(p: Person): string {
  const entries = Object.entries(p.meta);
  if (entries.length > 0) return entries.map(([k, v]) => `${k}: ${v}`).join(' · ');
  return `${p.num_faces} ${p.num_faces === 1 ? 'face' : 'faces'}`;
}

/* ------------------------------- Styles --------------------------------- */

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: {
    paddingHorizontal: spacing.lg,
    paddingBottom: TAB_BAR_SPACE,
    gap: spacing.md,
  },
  header: { gap: spacing.md, paddingTop: spacing.xs },

  // Enroll form
  form: { gap: spacing.md },
  formHint: {
    ...typography.caption,
    color: palette.textSecondary,
  },
  field: { gap: spacing.xs },
  fieldLabel: {
    ...typography.label,
    color: palette.textSecondary,
  },
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
  photoRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  photoBtn: { flexShrink: 1 },
  photoCount: {
    ...typography.caption,
    color: palette.accent,
    fontWeight: '700',
  },
  thumbStrip: { gap: spacing.sm, paddingVertical: spacing.xs },
  thumb: {
    width: 54,
    height: 54,
    borderRadius: radii.md,
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairline,
  },
  thumbMore: { alignItems: 'center', justifyContent: 'center' },
  thumbMoreText: { ...typography.bodyStrong, color: palette.textSecondary },

  // Person row
  list: { gap: spacing.md },
  personCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: radii.pill,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.surfaceElevated,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: palette.hairlineStrong,
  },
  avatarText: {
    ...typography.bodyStrong,
    color: palette.text,
  },
  personBody: { flex: 1, gap: 2 },
  personName: { ...typography.bodyStrong, color: palette.text },
  personMeta: { ...typography.caption, color: palette.textSecondary },
});
