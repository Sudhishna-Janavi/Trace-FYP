/**
 * The dashboard: what the money did, and what still needs a person.
 *
 * The web page puts four figures across the top, a cash-flow chart beneath
 * them and the ledger's most recent rows at the foot. A phone is a column,
 * so the four figures become a two-by-two grid and the chart keeps its full
 * width — but the order of the argument is unchanged, because the order is
 * the point: totals, then the shape over time, then where it went, then the
 * individual rows.
 */
import React from 'react';
import {
  RefreshControl, ScrollView, View, useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { Tx, api } from '../api';
import {
  Card, Divider, LayerChip, Loading, Money, Problem, Row, SectionTitle, T,
  ServerSvg, radius, space,
} from '../components/ui';
import { dateLabel } from '../theme';
import { useLoad } from '../useLoad';

export function DashboardScreen({ navigation }: any) {
  const { p } = useTheme();
  const { user } = useAuth();
  const { width } = useWindowDimensions();
  const state = useLoad(() => api.dashboard(), []);

  const chartWidth = width - space.lg * 2 - space.lg * 2;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <ScrollView
        contentContainerStyle={{ padding: space.lg, paddingBottom: 40 }}
        refreshControl={
          <RefreshControl refreshing={state.refreshing} onRefresh={state.refresh}
                          tintColor={p.muted} />
        }>

        {state.loading ? <Loading note="Loading your ledger…" /> : null}
        {state.error ? <Problem message={state.error} onRetry={state.reload} /> : null}

        {state.data?.has_data ? (() => {
          const d = state.data!;
          return (
            <>
              <T variant="display">{d.greeting}, {user?.display_name?.split(' ')[0]}</T>
              <T variant="small" tone="muted" style={{ marginTop: 6 }}>
                {d.totals.accounts} accounts · {d.totals.transactions.toLocaleString()} transactions
                {d.totals.first && d.totals.last
                  ? ` · ${dateLabel(d.totals.first)} – ${dateLabel(d.totals.last)}`
                  : ''}
              </T>

              <View style={{
                flexDirection: 'row', flexWrap: 'wrap', gap: space.md,
                marginTop: space.xl,
              }}>
                <Stat label="Net cash flow" value={d.totals.net} note="over 12 months" />
                <Stat label="Total spending" value={-d.totals.spending}
                      note="money out" />
                <Stat label="Total income" value={d.totals.income} note="credits only" />
                <Stat label="Needs review" count={d.totals.needs_review}
                      note="low confidence"
                      onPress={() => navigation.navigate('Review')} />
              </View>

              <SectionTitle>Monthly cash flow</SectionTitle>
              <Card>
                <ServerSvg xml={d.chart} width={chartWidth} />
                <View style={{ flexDirection: 'row', gap: space.lg, marginTop: space.md }}>
                  <Key colour={p.income} label="Income" />
                  <Key colour={p.spend} label="Spending" />
                </View>
              </Card>

              <SectionTitle>Spending by category</SectionTitle>
              <Card padded={false}>
                {d.categories.slice(0, 8).map((c, index) => (
                  <View key={c.category}>
                    {index ? <Divider /> : null}
                    <Row>
                      <View style={{
                        flexDirection: 'row', justifyContent: 'space-between',
                        alignItems: 'baseline',
                      }}>
                        <T variant="body">{c.category}</T>
                        <Money value={-c.amount} variant="body" decimals={0} />
                      </View>
                      <View style={{
                        height: 6, borderRadius: 3, backgroundColor: p.surface2,
                        marginTop: 8, overflow: 'hidden',
                      }}>
                        <View style={{
                          width: `${c.share}%`, height: '100%',
                          backgroundColor: p.accent,
                        }} />
                      </View>
                      <T variant="small" tone="faint" style={{ marginTop: 5 }}>
                        {c.share.toFixed(1)}% · {c.count} transactions
                      </T>
                    </Row>
                  </View>
                ))}
              </Card>

              <SectionTitle>Recent transactions</SectionTitle>
              <Card padded={false}>
                {d.recent.map((row, index) => (
                  <View key={row.id}>
                    {index ? <Divider /> : null}
                    <LedgerRow
                      row={row}
                      onPress={() => navigation.navigate('Transaction', { id: row.id })}
                    />
                  </View>
                ))}
              </Card>
            </>
          );
        })() : null}
      </ScrollView>
    </SafeAreaView>
  );
}

/* ------------------------------------------------------------------ bits -- */

function Stat({ label, value, count, note, onPress }: {
  label: string;
  value?: number;
  count?: number;
  note: string;
  onPress?: () => void;
}) {
  const { p } = useTheme();
  const { width } = useWindowDimensions();
  const boxWidth = (width - space.lg * 2 - space.md) / 2;
  return (
    <Card style={{ width: boxWidth }}>
      <T variant="label" tone="muted">{label}</T>
      <View style={{ marginTop: 6 }}>
        {value !== undefined
          ? <Money value={value} variant="title" decimals={0} />
          : <T variant="title" style={{ color: count ? p.review : p.ink }}>
              {count}
            </T>}
      </View>
      <T variant="small" tone="faint" style={{ marginTop: 4 }}>{note}</T>
    </Card>
  );
}

function Key({ colour, label }: { colour: string; label: string }) {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
      <View style={{ width: 9, height: 9, borderRadius: 2.5, backgroundColor: colour }} />
      <T variant="small" tone="muted">{label}</T>
    </View>
  );
}

/** One ledger row, shared by the dashboard, the ledger and the queue. */
export function LedgerRow({ row, onPress }: { row: Tx; onPress: () => void }) {
  return (
    <Row onPress={onPress}>
      <View style={{
        flexDirection: 'row', justifyContent: 'space-between',
        alignItems: 'flex-start',
      }}>
        <T variant="mono" numberOfLines={1} style={{ flex: 1, marginRight: space.md }}>
          {row.description}
        </T>
        <Money value={row.amount} variant="body" />
      </View>
      <View style={{
        flexDirection: 'row', alignItems: 'center',
        justifyContent: 'space-between', marginTop: 7,
      }}>
        {/* the date must give way rather than wrap: the chips beside it are
            fixed-width and a two-line date reads as a mistake */}
        <T variant="small" tone="faint" numberOfLines={1}
           style={{ flexShrink: 1, marginRight: space.sm }}>
          {dateLabel(row.date)} · {row.bank}
        </T>
        <View style={{ flexDirection: 'row', gap: 6, alignItems: 'center' }}>
          {row.needs_review ? <ReviewDot /> : null}
          <LayerChip layer={row.layer} label={row.category} />
        </View>
      </View>
    </Row>
  );
}

function ReviewDot() {
  const { p } = useTheme();
  return (
    <View style={{
      backgroundColor: p.reviewSoft, borderRadius: radius.pill,
      paddingHorizontal: 8, paddingVertical: 3,
    }}>
      <T variant="small" style={{ color: p.review, fontSize: 11 }}>Review</T>
    </View>
  );
}
