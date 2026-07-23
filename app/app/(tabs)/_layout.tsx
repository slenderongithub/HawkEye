/**
 * Bottom-tab navigator using the custom AnimatedTabBar. Five primary tabs;
 * Settings is a root-stack modal reached from each screen's header gear.
 */
import { Tabs } from 'expo-router';
import { AnimatedTabBar } from '@/components/AnimatedTabBar';

export default function TabsLayout() {
  return (
    <Tabs
      tabBar={(props) => <AnimatedTabBar {...props} />}
      screenOptions={{ headerShown: false }}
    >
      <Tabs.Screen name="live" />
      <Tabs.Screen name="incidents" />
      <Tabs.Screen name="attendance" />
      <Tabs.Screen name="people" />
      <Tabs.Screen name="drone" />
    </Tabs>
  );
}
