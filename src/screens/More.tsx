/**
 * Everything the tab bar has no room for.
 *
 * Five tabs is the most a thumb can reach without aiming, so Accounts and How
 * it works live here, alongside the account itself and the theme. The web
 * application makes the same split on a phone: four tabs and a More sheet.
 */
import React from 'react';
import { ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { IS_LOCAL } from '../config';
import {
  Button, Card, Divider, Row, SectionTitle, T, radius, space,
} from '../components/ui';
import { Icon, IconName } from '../components/Icon';

export function MoreScreen({ navigation }: any) {
  const { p, preference, setPreference } = useTheme();
  const { user, signOut } = useAuth();

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <ScrollView contentContainerStyle={{ padding: space.lg, paddingBottom: 40 }}>
        <T variant="display">More</T>

        <Card style={{ marginTop: space.lg }}>
          <T variant="heading">{user?.display_name}</T>
          <T variant="small" tone="muted" style={{ marginTop: 4 }}>
            {user?.email}
          </T>
        </Card>

        <SectionTitle>Pages</SectionTitle>
        <Card padded={false}>
          <Link icon="accounts" label="Accounts"
                note="Every institution, and what it contributed"
                onPress={() => navigation.navigate('Accounts')} />
          <Divider />
          <Link icon="how" label="How it works"
                note="The cascade, and how accurate it actually is"
                onPress={() => navigation.navigate('HowItWorks')} />
          <Divider />
          <Link icon="transactions" label="Import more statements"
                note="Add another export, or replace what is here"
                onPress={() => navigation.navigate('Connect')} />
        </Card>

        <SectionTitle>Appearance</SectionTitle>
        <View style={{ flexDirection: 'row', gap: space.sm }}>
          {(['light', 'dark', 'system'] as const).map(option => (
            <Button
              key={option}
              label={option === 'system' ? 'System' : option === 'light' ? 'Light' : 'Dark'}
              kind={preference === option ? 'primary' : 'ghost'}
              onPress={() => setPreference(option)}
              style={{ flex: 1 }}
            />
          ))}
        </View>
        <T variant="small" tone="faint" style={{ marginTop: space.sm }}>
          The choice is stored against your account, so it follows you to the
          browser.
        </T>

        <SectionTitle>Session</SectionTitle>
        <Button label="Sign out" kind="ghost" onPress={signOut} />

        <View style={{
          marginTop: space.xxl, padding: space.lg,
          backgroundColor: p.surface2, borderRadius: radius.card,
        }}>
          <T variant="small" tone="faint">
            Trace · CCDS25-1131 · a Final Year Project at the College of
            Computing and Data Science, Nanyang Technological University.
            {IS_LOCAL ? ' Connected to a development server.' : ''}
          </T>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function Link({ icon, label, note, onPress }: {
  icon: IconName; label: string; note: string; onPress: () => void;
}) {
  const { p } = useTheme();
  return (
    <Row onPress={onPress}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
        <Icon name={icon} colour={p.accent} size={20} />
        <View style={{ flex: 1 }}>
          <T variant="body">{label}</T>
          <T variant="small" tone="faint" style={{ marginTop: 2 }}>{note}</T>
        </View>
        <T variant="body" tone="faint">{'›'}</T>
      </View>
    </Row>
  );
}
