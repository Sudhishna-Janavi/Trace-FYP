/**
 * Sign in and create an account.
 *
 * One component for both, because the two forms differ by one field and a
 * verb. The web application does the same thing with one template, and the
 * wording below is taken from it so a person who has used Trace in a browser
 * reads the same sentences here.
 */
import React, { useState } from 'react';
import {
  KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { ApiError } from '../api';
import { Button, Card, Field, Problem, T, radius, space } from '../components/ui';

type Mode = 'signin' | 'signup';

export function AuthScreen() {
  const { p } = useTheme();
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<Mode>('signin');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isSignUp = mode === 'signup';

  async function submit() {
    setError(null);
    setBusy(true);
    try {
      if (isSignUp) await signUp(name, email, password);
      else await signIn(email, password);
    } catch (problem) {
      setError(problem instanceof ApiError
        ? problem.message
        : 'Something went wrong. Please try again.');
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView
          contentContainerStyle={{
            flexGrow: 1,
            justifyContent: 'center',
            padding: space.xl,
          }}
          keyboardShouldPersistTaps="handled">

          {/* the brand mark: a rounded square in the accent, as on the web */}
          <View style={{
            flexDirection: 'row', alignItems: 'center', gap: 9,
            marginBottom: space.xl,
          }}>
            <View style={{
              width: 22, height: 22, borderRadius: 7, backgroundColor: p.accent,
            }} />
            <T variant="heading" style={{ fontSize: 17 }}>Trace</T>
          </View>

          <Card>
            <T variant="title">
              {isSignUp ? 'Create your account' : 'Sign in'}
            </T>
            <T variant="small" tone="muted" style={{ marginTop: 6 }}>
              {isSignUp
                ? 'Trace categorises your transactions and shows you why.'
                : 'Welcome back.'}
            </T>

            {error ? (
              <View style={{ marginTop: space.lg }}>
                <Problem message={error} />
              </View>
            ) : null}

            {isSignUp ? (
              <Field
                label="Your name"
                value={name}
                onChangeText={setName}
                autoCapitalize="words"
                autoComplete="name"
                textContentType="name"
                returnKeyType="next"
              />
            ) : null}

            <Field
              label="Email"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="email-address"
              autoComplete="email"
              textContentType="emailAddress"
              returnKeyType="next"
            />

            <Field
              label="Password"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
              autoCapitalize="none"
              autoComplete={isSignUp ? 'new-password' : 'current-password'}
              textContentType={isSignUp ? 'newPassword' : 'password'}
              returnKeyType="go"
              onSubmitEditing={submit}
            />

            {isSignUp ? (
              <T variant="small" tone="faint" style={{ marginTop: 8 }}>
                At least eight characters.
              </T>
            ) : null}

            <Button
              label={isSignUp ? 'Create account' : 'Sign in'}
              onPress={submit}
              busy={busy}
              style={{ marginTop: space.xl }}
            />
          </Card>

          <Pressable
            onPress={() => { setMode(isSignUp ? 'signin' : 'signup'); setError(null); }}
            style={{ marginTop: space.xl, alignItems: 'center', minHeight: 46,
                     justifyContent: 'center' }}>
            <T variant="small" tone="muted">
              {isSignUp ? 'Already have an account? ' : 'New here? '}
              <T variant="small" tone="accent">
                {isSignUp ? 'Sign in' : 'Create an account'}
              </T>
            </T>
          </Pressable>

          <View style={{
            marginTop: space.xxl,
            padding: space.lg,
            borderRadius: radius.card,
            backgroundColor: p.surface2,
            borderColor: p.line,
            borderWidth: StyleSheet.hairlineWidth,
          }}>
            <T variant="small" tone="muted">
              Connecting a real account requires an agreement with each bank
              and your explicit consent through an open-banking API. That
              access is not available to a student project, so Trace
              demonstrates the same ingestion path using exported statement
              files.
            </T>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
