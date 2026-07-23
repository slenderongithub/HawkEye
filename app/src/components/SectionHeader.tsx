/** Uppercase section label with an optional right-aligned count/accessory. */
import React from 'react';
import { Text, View, StyleSheet } from 'react-native';
import { palette, spacing, typography } from '@/theme';

type Props = {
  title: string;
  count?: number;
  right?: React.ReactNode;
};

export function SectionHeader({ title, count, right }: Props) {
  return (
    <View style={styles.row}>
      <Text style={styles.title}>
        {title}
        {count !== undefined ? <Text style={styles.count}>  {count}</Text> : null}
      </Text>
      {right}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
    marginTop: spacing.xs,
  },
  title: {
    ...typography.label,
    color: palette.textSecondary,
  },
  count: {
    ...typography.label,
    color: palette.textTertiary,
  },
});
