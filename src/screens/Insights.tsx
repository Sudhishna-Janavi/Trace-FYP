/**
 * Insights: what the spending actually did, said plainly.
 *
 * Nothing here is a score. A single number claiming to rate someone's
 * financial health would need a threshold, and every threshold available is
 * invented rather than derived — so these are statements the data supports
 * and nothing more. The methodology is on the screen, at the bottom, for the
 * same reason.
 */
import React from 'react';
import {
  RefreshControl, ScrollView, View, useWindowDimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useTheme } from '../ThemeContext';
import { api } from '../api';
import {
  Card, Divider, Loading, Money, Pill, Problem, Row, SectionTitle, ServerSvg,
  T, space,
} from '../components/ui';
import { dateLabel, money } from '../theme';
import { useLoad } from '../useLoad';

export function InsightsScreen() {
  const { p } = useTheme();
  const { width } = useWindowDimensions();
  const state = useLoad(() => api.insights(), []);
  const inner = width - space.lg * 4;
  const d = state.data;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <ScrollView
        contentContainerStyle={{ padding: space.lg, paddingBottom: 40 }}
        refreshControl={
          <RefreshControl refreshing={state.refreshing} onRefresh={state.refresh}
                          tintColor={p.muted} />
        }>
        <T variant="display">Insights</T>

        {state.loading ? <Loading /> : null}
        {state.error ? <Problem message={state.error} onRetry={state.reload} /> : null}

        {d?.has_data ? (
          <>
            <SectionTitle>Where the money went</SectionTitle>
            <Card>
              <View style={{ alignItems: 'center' }}>
                <ServerSvg xml={d.wheel} width={Math.min(inner, 260)} />
              </View>
              <View style={{ marginTop: space.lg, gap: 10 }}>
                {/* the swatch carries the colour the server gave each
                    segment, so a slice can be matched to its row; without it
                    the wheel is decoration rather than a key */}
                {d.segments.map(seg => (
                  <View key={seg.category} style={{
                    flexDirection: 'row', alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <View style={{
                      flexDirection: 'row', alignItems: 'center', gap: 8,
                      flexShrink: 1,
                    }}>
                      <View style={{
                        width: 10, height: 10, borderRadius: 2.5,
                        backgroundColor: seg.colour ?? p.accent,
                      }} />
                      <T variant="small" numberOfLines={1}>{seg.category}</T>
                    </View>
                    <T variant="small" tone="muted">
                      {money(seg.amount, 0)} · {seg.share.toFixed(1)}%
                    </T>
                  </View>
                ))}
              </View>
            </Card>

            {d.change.available ? (
              <>
                <SectionTitle
                  note={`${d.change.month} against your ${d.change.months}-month average.`}>
                  What changed
                </SectionTitle>
                <Card>
                  <ServerSvg xml={d.change_chart} width={inner} />
                  <View style={{ marginTop: space.lg, gap: 10 }}>
                    {d.change.rows.slice(0, 6).map(row => (
                      <View key={row.category} style={{
                        flexDirection: 'row', justifyContent: 'space-between',
                        alignItems: 'center',
                      }}>
                        <T variant="small">{row.category}</T>
                        <Money value={-row.vs_average} variant="small" decimals={0} />
                      </View>
                    ))}
                  </View>
                </Card>
              </>
            ) : null}

            {d.signals ? (
              <>
                <SectionTitle note="Descriptive facts, not a rating.">
                  Signals
                </SectionTitle>
                <Card padded={false}>
                  <Fact label="Months of data" value={String(d.signals.months)} />
                  <Divider />
                  <Fact label="Lowest month" value={money(d.signals.lowest, 0)} />
                  <Divider />
                  <Fact label="Highest month" value={money(d.signals.highest, 0)} />
                  <Divider />
                  <Fact label="Average month" value={money(d.signals.average, 0)} />
                  <Divider />
                  <Fact
                    label="Trend"
                    value={`${d.signals.trend_per_month >= 0 ? '+' : '−'}${money(Math.abs(d.signals.trend_per_month), 0).replace('$', '$')} a month`}
                  />
                  <Divider />
                  <Fact
                    label="Months spending exceeded income"
                    value={String(d.signals.overspent_months)}
                  />
                </Card>
              </>
            ) : null}

            {d.benchmark.length ? (
              <>
                <SectionTitle
                  note="Against the US Bureau of Labor Statistics household average. A comparison, not a judgement.">
                  Benchmark
                </SectionTitle>
                {d.is_demo ? (
                  <View style={{ marginBottom: space.md }}>
                    <Pill
                      label="Demonstration amounts were calibrated to this same survey"
                      tone="review"
                    />
                  </View>
                ) : null}
                <Card padded={false}>
                  {d.benchmark.slice(0, 8).map((b, index) => (
                    <View key={b.category}>
                      {index ? <Divider /> : null}
                      <Row>
                        <View style={{
                          flexDirection: 'row', justifyContent: 'space-between',
                        }}>
                          <T variant="small">{b.category}</T>
                          <T variant="small" tone={b.gap > 0 ? 'spend' : 'muted'}>
                            {b.share.toFixed(1)}% vs {b.benchmark.toFixed(1)}%
                          </T>
                        </View>
                      </Row>
                    </View>
                  ))}
                </Card>
              </>
            ) : null}

            {d.unusual.length ? (
              <>
                <SectionTitle note="Far above the usual for their kind of spending, and not recurring.">
                  Unusual
                </SectionTitle>
                <Card padded={false}>
                  {d.unusual.map((u, index) => (
                    <View key={u.id}>
                      {index ? <Divider /> : null}
                      <Row>
                        <View style={{
                          flexDirection: 'row', justifyContent: 'space-between',
                        }}>
                          <T variant="monoSmall" numberOfLines={1}
                             style={{ flex: 1, marginRight: space.md }}>
                            {u.description}
                          </T>
                          <Money value={-u.amount} variant="small" />
                        </View>
                        <T variant="small" tone="faint" style={{ marginTop: 3 }}>
                          {dateLabel(u.date)} · usual for {u.category} is about{' '}
                          {money(u.baseline, 0)}
                        </T>
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

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <Row>
      <View style={{
        flexDirection: 'row', justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <T variant="small" tone="muted" style={{ flex: 1, marginRight: space.md }}>
          {label}
        </T>
        <T variant="small">{value}</T>
      </View>
    </Row>
  );
}
