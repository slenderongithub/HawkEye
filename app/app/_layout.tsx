/**
 * Root layout: global providers (react-query, gesture handler, safe area),
 * settings hydration, the single live WebSocket connection, the toast host, and
 * the root navigation stack. Everything else renders under this.
 */
import { useEffect } from 'react';
import { StyleSheet } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import * as SystemUI from 'expo-system-ui';
import { Stack } from 'expo-router';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { palette } from '@/theme';
import { useSettings } from '@/stores/settings';
import { useLiveSocket } from '@/ws/useLiveSocket';
import { ToastHost } from '@/components/ToastHost';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
});

SystemUI.setBackgroundColorAsync(palette.bg).catch(() => {});

function AppShell() {
  const hydrate = useSettings((s) => s.hydrate);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  // Single shared WebSocket for live frames + incident/attendance/status.
  useLiveSocket();

  return (
    <>
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: palette.bg },
          animation: 'fade',
        }}
      >
        <Stack.Screen name="(tabs)" />
        <Stack.Screen
          name="incident/[id]"
          options={{ presentation: 'card', animation: 'slide_from_right' }}
        />
        <Stack.Screen
          name="settings"
          options={{ presentation: 'modal', animation: 'slide_from_bottom' }}
        />
      </Stack>
      <ToastHost />
      <StatusBar style="light" />
    </>
  );
}

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={styles.root}>
      <SafeAreaProvider>
        <QueryClientProvider client={queryClient}>
          <AppShell />
        </QueryClientProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: palette.bg },
});
