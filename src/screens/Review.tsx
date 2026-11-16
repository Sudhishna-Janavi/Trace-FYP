/**
 * The review queue.
 *
 * The gate at τ = 0.60 exists so the model declines rather than guesses. This
 * is where the declined rows are handed to the person who actually knows the
 * answer, ordered by confidence so the least certain is dealt with first.
 *
 * Every row can be settled without leaving the list — confirming is one tap,
 * and the row disappears. Opening the explanation is a tap on the row itself,
 * for the cases where the answer is not obvious.
 */
import React from 'react';
import { RefreshControl, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { ApiError, api } from '../api';
import {
  Button, Card, Divider, Loading, Money, Problem, Row, SectionTitle, T,
  radius, space,
} from '../components/ui';
import { dateLabel } from '../theme';
import { useLoad } from '../useLoad';

export function ReviewScreen({ navigation }: any) {
  const { p } = useTheme();
  const { refresh } = useAuth();
  const state = useLoad(() => api.review(), []);
  const [busy, setBusy] = React.useState<number | null>(null);
  const [problem, setProblem] = React.useState<string | null>(null);

  async function confirm(id: number) {
    setProblem(null);
    setBusy(id);
    try {
      await api.confirm(id);
      state.set(state.data
        ? { ...state.data, rows: state.data.rows.filter(r => r.id !== id),
            reviewed: state.data.reviewed + 1 }
        : state.data as any);
      refresh();
    } catch (error) {
      setProblem(error instanceof ApiError ? error.message : 'That did not save.');
    } finally {
      setBusy(null);
    }
  }

  const d = state.data;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <ScrollView
        contentContainerStyle={{ padding: space.lg, paddingBottom: 40 }}
        refreshControl={
          <RefreshControl refreshing={state.refreshing} onRefresh={state.refresh}
                          tintColor={p.muted} />
        }>
        <T variant="display">Review</T>
        <T variant="small" tone="muted" style={{ marginTop: 6 }}>
          Trace holds a transaction back when its confidence falls below{' '}
          {d ? Math.round(d.threshold * 100) : 60}%. These are the ones it
          would rather you decided.
        </T>

        {state.loading ? <Loading /> : null}
        {state.error ? <Problem message={state.error} onRetry={state.reload} /> : null}
        {problem ? (
          <View style={{ marginTop: space.lg }}>
            <Problem message={problem} />
          </View>
        ) : null}

        {d ? (
          <>
            <View style={{ flexDirection: 'row', gap: space.md, marginTop: space.lg }}>
              <Card style={{ flex: 1 }}>
                <T variant="label" tone="muted">Waiting</T>
                <T variant="title" style={{
                  marginTop: 6, color: d.rows.length ? p.review : p.accent,
                }}>
                  {d.rows.length}
                </T>
              </Card>
              <Card style={{ flex: 1 }}>
                <T variant="label" tone="muted">Settled</T>
                <T variant="title" style={{ marginTop: 6 }}>{d.reviewed}</T>
              </Card>
            </View>

            {d.rows.length === 0 ? (
              <Card tinted style={{ marginTop: space.xl }}>
                <T variant="heading">Nothing is waiting</T>
                <T variant="small" tone="muted" style={{ marginTop: 6 }}>
                  Every transaction the cascade was unsure about has been
                  settled. New ones will appear here after your next import.
                </T>
              </Card>
            ) : (
              <>
                <SectionTitle note="Least confident first.">
                  Waiting for you
                </SectionTitle>
                <Card padded={false}>
                  {d.rows.map((row, index) => (
                    <View key={row.id}>
                      {index ? <Divider /> : null}
                      <Row onPress={() => navigation.navigate('Transaction', { id: row.id })}>
                        <View style={{
                          flexDirection: 'row', justifyContent: 'space-between',
                          alignItems: 'flex-start',
                        }}>
                          <T variant="mono" numberOfLines={1}
                             style={{ flex: 1, marginRight: space.md }}>
                            {row.description}
                          </T>
                          <Money value={row.amount} variant="body" />
                        </View>

                        <View style={{
                          flexDirection: 'row', alignItems: 'center',
                          gap: 8, marginTop: 7,
                        }}>
                          <View style={{
                            backgroundColor: p.reviewSoft,
                            borderRadius: radius.pill,
                            paddingHorizontal: 9, paddingVertical: 4,
                          }}>
                            <T variant="label" style={{ color: p.review }}>
                              {Math.round(row.confidence * 100)}% · {row.category}
                            </T>
                          </View>
                          <T variant="small" tone="faint">
                            {dateLabel(row.date)} · {row.bank}
                          </T>
                        </View>

                        <View style={{ flexDirection: 'row', gap: space.sm,
                                       marginTop: space.md }}>
                          {/* the category is already in the pill above, so
                              the button stays one short word and the two
                              buttons keep the same height whatever it is */}
                          <Button
                            label="Confirm"
                            kind="ghost"
                            busy={busy === row.id}
                            onPress={() => confirm(row.id)}
                            style={{ flex: 1 }}
                          />
                          <Button
                            label="Change"
                            kind="quiet"
                            onPress={() => navigation.navigate('Transaction', { id: row.id })}
                            style={{ flex: 1 }}
                          />
                        </View>
                      </Row>
                    </View>
                  ))}
                </Card>
              </>
            )}

            {d.corrections.length ? (
              <>
                <SectionTitle note="Checked before any rule or model, for every future import.">
                  What Trace has learnt from you
                </SectionTitle>
                <Card padded={false}>
                  {d.corrections.map((c, index) => (
                    <View key={c.merchant}>
                      {index ? <Divider /> : null}
                      <Row>
                        <View style={{
                          flexDirection: 'row', justifyContent: 'space-between',
                          alignItems: 'center',
                        }}>
                          <T variant="monoSmall" tone="muted" numberOfLines={1}
                             style={{ flex: 1, marginRight: space.md }}>
                            {c.merchant}
                          </T>
                          <T variant="label" style={{ color: p.accent }}>
                            {c.category}
                          </T>
                        </View>
                        {c.times_applied ? (
                          <T variant="small" tone="faint" style={{ marginTop: 4 }}>
                            applied to {c.times_applied}{' '}
                            {c.times_applied === 1 ? 'transaction' : 'transactions'}
                          </T>
                        ) : null}
                      </Row>
                    </View>
                  ))}
                </Card>
              </>
            ) : null}
          </>
        ) : null}
      </ScrollView>
    </SafeAreaView>
  );
}
