/** Screen shell: background color + safe-area insets. */
import React from 'react';
import { View, StyleSheet, type StyleProp, type ViewStyle } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { palette } from '@/theme';

type Props = {
  children: React.ReactNode;
  /** Apply top safe-area inset (screens with their own header don't need it). */
  edges?: { top?: boolean; bottom?: boolean };
  style?: StyleProp<ViewStyle>;
};

export function Screen({ children, edges = { top: true }, style }: Props) {
  const insets = useSafeAreaInsets();
  return (
    <View
      style={[
        styles.root,
        {
          paddingTop: edges.top ? insets.top : 0,
          paddingBottom: edges.bottom ? insets.bottom : 0,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: palette.bg,
  },
});
