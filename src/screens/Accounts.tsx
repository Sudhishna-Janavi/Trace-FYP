/**
 * The institutions behind the ledger.
 *
 * Worth a screen of its own because the export conventions are the problem
 * the ingestion layer exists to solve, and naming them makes that visible:
 * one signed amount column, a debit/credit pair, twelve monthly files with
 * the sign inverted. The ledger reads as one list precisely because these
 * three do not.
 */
import React from 'react';
import { RefreshControl, ScrollView, View } from 'react-native';

import { useTheme } from '../ThemeContext';
import { api } from '../api';
import {
  Card, Divider, Loading, Money, Problem, Row, T, space,
} from '../components/ui';
import { dateLabel } from '../theme';
import { useLoad } from '../useLoad';

export function AccountsScreen() {
  const { p } = useTheme();
  const state = useLoad(() => api.accounts(), []);
  const d = state.data;

  return (
    <ScrollView
      style={{ backgroundColor: p.bg }}
      contentContainerStyle={{ padding: space.lg, paddingBottom: 40 }}
      refreshControl={
        <RefreshControl refreshing={state.refreshing} onRefresh={state.refresh}
                        tintColor={p.muted} />
      }>
      {state.loading ? <Loading /> : null}
      {state.error ? <Problem message={state.error} onRetry={state.reload} /> : null}

      {d ? (
        <>
          <T variant="small" tone="muted">
            {d.detail.length} {d.detail.length === 1 ? 'institution' : 'institutions'} ·{' '}
            {d.total.toLocaleString()} transactions in one ledger
          </T>

          <View style={{ gap: space.lg, marginTop: space.lg }}>
            {d.detail.map(item => (
              <Card key={item.account.id} padded={false}>
                <Row>
                  <View style={{
                    flexDirection: 'row', justifyContent: 'space-between',
                    alignItems: 'baseline',
                  }}>
                    <T variant="heading">{item.account.bank}</T>
                    <Money value={-item.spend} variant="body" decimals={0} />
                  </View>
                  <T variant="small" tone="faint" style={{ marginTop: 3 }}>
                    {item.account.name} · {item.share.toFixed(1)}% of spending
                  </T>
                </Row>
                <Divider />
                <Row>
                  <T variant="label" tone="muted">Export format</T>
                  <T variant="small" style={{ marginTop: 4 }}>
                    {item.account.source_format}
                  </T>
                </Row>
                <Divider />
                <Row>
                  <View style={{
                    flexDirection: 'row', justifyContent: 'space-between',
                  }}>
                    <T variant="small" tone="muted">Transactions</T>
                    <T variant="small">{item.rows.toLocaleString()}</T>
                  </View>
                  {item.first && item.last ? (
                    <View style={{
                      flexDirection: 'row', justifyContent: 'space-between',
                      marginTop: 6,
                    }}>
                      <T variant="small" tone="muted">Covering</T>
                      <T variant="small">
                        {dateLabel(item.first)} – {dateLabel(item.last)}
                      </T>
                    </View>
                  ) : null}
                </Row>
              </Card>
            ))}
          </View>

          <Card tinted style={{ marginTop: space.xl }}>
            <T variant="small" tone="muted">
              Each institution has its own adapter, which turns its export into
              one canonical record. Nothing downstream — not the cascade, not
              the analytics, not this screen — knows which bank a transaction
              came from except as a label.
            </T>
          </Card>
        </>
      ) : null}
    </ScrollView>
  );
}
