/**
 * Getting transactions into Trace.
 *
 * Two routes, the same two the web application offers: statement files the
 * person chooses, or the bundled three-bank demonstration set. Uploading is
 * presented first because it is the one that uses their own data.
 *
 * The picker is deliberately not restricted by file type. A bank's export
 * arrives with whatever uniform type identifier that bank happened to set,
 * and filtering on type greys out perfectly valid files on iOS. The server
 * recognises a file by its columns rather than its name or its type, so
 * letting anything through and rejecting it with a reason is both more
 * robust and more honest than pretending to know in advance.
 */
import * as DocumentPicker from 'expo-document-picker';
import React, { useState } from 'react';
import { ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { ApiError, IngestSummary, api } from '../api';
import {
  Button, Card, Divider, Pill, Problem, T, radius, space,
} from '../components/ui';
import { layerColour } from '../theme';

interface Chosen {
  uri: string;
  name: string;
  size?: number;
  mimeType?: string;
}

export function ConnectScreen() {
  const { p } = useTheme();
  const { markConnected } = useAuth();
  const [chosen, setChosen] = useState<Chosen[]>([]);
  const [busy, setBusy] = useState<null | 'upload' | 'demo'>(null);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<IngestSummary | null>(null);

  async function pick() {
    setError(null);
    const result = await DocumentPicker.getDocumentAsync({
      multiple: true,
      copyToCacheDirectory: true,
    });
    if (result.canceled) return;
    setChosen(result.assets.map(a => ({
      uri: a.uri, name: a.name, size: a.size ?? undefined,
      mimeType: a.mimeType ?? undefined,
    })));
  }

  async function run(what: 'upload' | 'demo') {
    setError(null);
    setBusy(what);
    try {
      const result = what === 'upload'
        ? await api.upload(chosen)
        : await api.loadDemo();
      // The summary is shown first and `markConnected` deliberately waits for
      // the reader to leave it. Marking the account connected here would flip
      // the shell to the tab bar immediately and the import — which layer
      // answered how much of it — would never be seen.
      setSummary(result);
    } catch (problem) {
      setError(problem instanceof ApiError
        ? problem.message
        : 'That import could not be completed.');
    } finally {
      setBusy(null);
    }
  }

  const bytes = chosen.reduce((sum, f) => sum + (f.size ?? 0), 0);
  const sizeLabel = bytes > 1048576
    ? `${(bytes / 1048576).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;

  if (summary) {
    return (
      <Summary
        summary={summary}
        onDone={() => { setSummary(null); markConnected(); }}
      />
    );
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <ScrollView contentContainerStyle={{ padding: space.lg, paddingBottom: 48 }}>
        <T variant="display">Connect your accounts</T>
        <T variant="small" tone="muted" style={{ marginTop: 8 }}>
          Trace reads exported bank statements, normalises them into one
          format and categorises every transaction — telling you why each
          decision was made.
        </T>

        {error ? (
          <View style={{ marginTop: space.lg }}>
            <Problem message={error} />
          </View>
        ) : null}

        <Card style={{ marginTop: space.xl }}>
          <View style={{
            flexDirection: 'row', alignItems: 'center',
            justifyContent: 'space-between', marginBottom: 8,
          }}>
            <T variant="heading">Upload your statements</T>
            <Pill label="Recommended" tone="accent" />
          </View>
          <T variant="small" tone="muted">
            CSV exports from Chase, Capital One or Apple Card. Choose as many
            as you like, or a .zip containing them. Each file is recognised by
            its columns rather than its name, so what your bank called it does
            not matter.
          </T>

          <Button
            label={chosen.length ? 'Choose different files' : 'Choose files'}
            kind="ghost"
            onPress={pick}
            style={{ marginTop: space.lg }}
          />

          {chosen.length ? (
            <View style={{
              marginTop: space.md,
              padding: space.md,
              backgroundColor: p.surface2,
              borderRadius: radius.card,
            }}>
              <T variant="small" tone="muted">
                {chosen.length} {chosen.length === 1 ? 'file' : 'files'} · {sizeLabel}
              </T>
              <T variant="monoSmall" tone="faint" numberOfLines={3}
                 style={{ marginTop: 4 }}>
                {chosen.slice(0, 4).map(f => f.name).join(', ')}
                {chosen.length > 4 ? ', …' : ''}
              </T>
            </View>
          ) : null}

          {chosen.length ? (
            <Button
              label={busy === 'upload'
                ? 'Reading and categorising…'
                : `Upload and categorise ${chosen.length} ${chosen.length === 1 ? 'file' : 'files'}`}
              onPress={() => run('upload')}
              busy={busy === 'upload'}
              disabled={busy !== null}
              style={{ marginTop: space.md }}
            />
          ) : null}
        </Card>

        <Card tinted style={{ marginTop: space.lg }}>
          <T variant="heading">Use the demonstration set</T>
          <T variant="small" tone="muted" style={{ marginTop: 8 }}>
            Loads three accounts — Chase, Capital One and Apple Card —
            exported in three different formats, and runs them through the
            full pipeline.
          </T>
          <Button
            label={busy === 'demo' ? 'Reading and categorising…' : 'Load demonstration data'}
            kind="ghost"
            onPress={() => run('demo')}
            busy={busy === 'demo'}
            disabled={busy !== null}
            style={{ marginTop: space.lg }}
          />
        </Card>

        <T variant="small" tone="faint" style={{ marginTop: space.xl }}>
          Connecting a real account requires an agreement with each bank and
          your explicit consent through an open-banking API. That access is
          not available to a student project, so Trace demonstrates the same
          ingestion path using exported statement files.
        </T>
      </ScrollView>
    </SafeAreaView>
  );
}

/* -------------------------------------------------- what the import did -- */

function Summary({ summary, onDone }: { summary: IngestSummary; onDone: () => void }) {
  const { p } = useTheme();
  const total = summary.layers.reduce((sum, l) => sum + l.count, 0) || 1;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <ScrollView contentContainerStyle={{ padding: space.lg, paddingBottom: 48 }}>
        <Pill label="Imported" tone="accent" />
        <T variant="display" style={{ marginTop: space.md }}>
          {summary.transactions.toLocaleString()} transactions
        </T>
        <T variant="small" tone="muted" style={{ marginTop: 8 }}>
          Across {summary.accounts} {summary.accounts === 1 ? 'account' : 'accounts'}
          {summary.uploaded
            ? ` from ${summary.uploaded.banks_phrase}, read from ${summary.uploaded.files} ${summary.uploaded.files === 1 ? 'file' : 'files'}`
            : ''}.
          {' '}{summary.needs_review} held for your review.
        </T>

        {summary.uploaded?.rejected?.length ? (
          <Card tinted style={{ marginTop: space.lg }}>
            <T variant="heading" style={{ fontSize: 14 }}>
              {summary.uploaded.rejected.length}{' '}
              {summary.uploaded.rejected.length === 1 ? 'file was' : 'files were'} not recognised
            </T>
            {summary.uploaded.rejected.slice(0, 4).map(r => (
              <T key={r.name} variant="small" tone="muted" style={{ marginTop: 6 }}>
                {r.name} — {r.why}
              </T>
            ))}
          </Card>
        ) : null}

        <Card style={{ marginTop: space.xl, padding: 0 }}>
          <View style={{ padding: space.lg }}>
            <T variant="heading">Which layer decided</T>
            <T variant="small" tone="muted" style={{ marginTop: 6 }}>
              Each layer answers only what it can account for, and passes the
              rest down.
            </T>
          </View>
          <Divider />
          <View style={{ padding: space.lg, gap: space.lg }}>
            {summary.layers.map(layer => (
              <View key={layer.key}>
                <View style={{
                  flexDirection: 'row', justifyContent: 'space-between',
                  alignItems: 'baseline', marginBottom: 6,
                }}>
                  <T variant="label">{layer.label}</T>
                  <T variant="small" tone="muted">
                    {layer.count.toLocaleString()} · {Math.round(layer.count / total * 100)}%
                  </T>
                </View>
                <View style={{
                  height: 8, borderRadius: 4, backgroundColor: p.surface2,
                  overflow: 'hidden',
                }}>
                  <View style={{
                    width: `${(layer.count / total) * 100}%`,
                    height: '100%',
                    backgroundColor: layerColour(layer.key, p),
                  }} />
                </View>
                <T variant="small" tone="faint" style={{ marginTop: 5 }}>
                  {layer.note}
                </T>
              </View>
            ))}
          </View>
        </Card>

        <Button label="See your dashboard" onPress={onDone}
                style={{ marginTop: space.xl }} />
      </ScrollView>
    </SafeAreaView>
  );
}
