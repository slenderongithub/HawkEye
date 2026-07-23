import { Redirect } from 'expo-router';

/** Entry point → the Live monitoring screen. */
export default function Index() {
  return <Redirect href="/(tabs)/live" />;
}
