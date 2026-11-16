/**
 * Trace — the native client.
 *
 * Three gates, in order: the fonts must be present before anything is drawn,
 * there must be someone signed in, and that person must have transactions.
 * Only past all three is the tab bar shown. The web application makes the
 * same three decisions with redirects; here they are states, which is the
 * honest way to express them when there is no address bar to redirect.
 *
 * The navigation is a native stack over a native tab bar — a real
 * UITabBarController with a real UINavigationController inside it — so the
 * gestures, the transitions and the way a screen is dismissed are the
 * platform's, not an imitation of them.
 */
import {
  IBMPlexMono_400Regular,
} from '@expo-google-fonts/ibm-plex-mono';
import {
  IBMPlexSans_400Regular, IBMPlexSans_500Medium, IBMPlexSans_600SemiBold,
  IBMPlexSans_700Bold, useFonts,
} from '@expo-google-fonts/ibm-plex-sans';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { ActivityIndicator, Text, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { AuthProvider, useAuth } from './src/AuthContext';
import { ThemeProvider, useTheme } from './src/ThemeContext';
import { Icon } from './src/components/Icon';
import { AuthScreen } from './src/screens/Auth';
import { ConnectScreen } from './src/screens/Connect';
import { DashboardScreen } from './src/screens/Dashboard';
import { TransactionsScreen } from './src/screens/Transactions';
import { TransactionScreen } from './src/screens/Transaction';
import { ReviewScreen } from './src/screens/Review';
import { InsightsScreen } from './src/screens/Insights';
import { MoreScreen } from './src/screens/More';
import { AccountsScreen } from './src/screens/Accounts';
import { HowItWorksScreen } from './src/screens/HowItWorks';
import { type } from './src/theme';

const Tabs = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

function TabBar() {
  const { p } = useTheme();
  const { reviewCount } = useAuth();

  return (
    <Tabs.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: p.accent,
        tabBarInactiveTintColor: p.faint,
        tabBarStyle: {
          backgroundColor: p.surface,
          borderTopColor: p.line,
        },
        tabBarLabelStyle: { fontFamily: type.label.fontFamily, fontSize: 11 },
        tabBarIcon: ({ color }) => {
          const icons: Record<string, any> = {
            Dashboard: 'dashboard', Transactions: 'transactions',
            Review: 'review', Insights: 'insights', More: 'more',
          };
          return <Icon name={icons[route.name]} colour={color} />;
        },
      })}>
      <Tabs.Screen name="Dashboard" component={DashboardScreen} />
      <Tabs.Screen name="Transactions" component={TransactionsScreen} />
      <Tabs.Screen
        name="Review"
        component={ReviewScreen}
        options={{ tabBarBadge: reviewCount || undefined }}
      />
      <Tabs.Screen name="Insights" component={InsightsScreen} />
      <Tabs.Screen name="More" component={MoreScreen} />
    </Tabs.Navigator>
  );
}

function Shell() {
  const { p, scheme } = useTheme();
  const { ready, user, hasData } = useAuth();

  if (!ready) {
    return (
      <View style={{ flex: 1, backgroundColor: p.bg, alignItems: 'center',
                     justifyContent: 'center' }}>
        <ActivityIndicator color={p.accent} />
      </View>
    );
  }

  if (!user) return <AuthScreen />;
  if (!hasData) return <ConnectScreen />;

  const navTheme = {
    ...(scheme === 'dark' ? DarkTheme : DefaultTheme),
    colors: {
      ...(scheme === 'dark' ? DarkTheme : DefaultTheme).colors,
      background: p.bg,
      card: p.surface,
      text: p.ink,
      border: p.line,
      primary: p.accent,
    },
  };

  return (
    <NavigationContainer theme={navTheme}>
      <Stack.Navigator
        screenOptions={{
          headerStyle: { backgroundColor: p.surface },
          headerTitleStyle: { ...type.heading, color: p.ink } as any,
          headerTintColor: p.accent,
          contentStyle: { backgroundColor: p.bg },
        }}>
        <Stack.Screen name="Tabs" component={TabBar} options={{ headerShown: false }} />
        <Stack.Screen
          name="Transaction"
          component={TransactionScreen}
          options={{ title: 'Why this category', presentation: 'modal' }}
        />
        <Stack.Screen name="Accounts" component={AccountsScreen}
                      options={{ title: 'Accounts' }} />
        <Stack.Screen name="HowItWorks" component={HowItWorksScreen}
                      options={{ title: 'How it works' }} />
        <Stack.Screen name="Connect" component={ConnectScreen}
                      options={{ title: 'Connect accounts' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

function Themed() {
  const { user } = useAuth();
  return (
    <ThemeProvider initial={user?.theme ?? 'system'}>
      <Inner />
    </ThemeProvider>
  );
}

function Inner() {
  const { scheme } = useTheme();
  return (
    <>
      <StatusBar style={scheme === 'dark' ? 'light' : 'dark'} />
      <Shell />
    </>
  );
}

export default function App() {
  const [fontsReady] = useFonts({
    IBMPlexSans_400Regular,
    IBMPlexSans_500Medium,
    IBMPlexSans_600SemiBold,
    IBMPlexSans_700Bold,
    IBMPlexMono_400Regular,
  });

  if (!fontsReady) {
    // Drawing in a substitute face and reflowing once the real one loads is
    // worse than a moment of nothing, so nothing is drawn.
    return <View style={{ flex: 1, backgroundColor: '#FBFBFA' }} />;
  }

  return (
    <SafeAreaProvider>
      <AuthProvider>
        <Themed />
      </AuthProvider>
    </SafeAreaProvider>
  );
}
