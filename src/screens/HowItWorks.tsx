/**
 * The cascade, and how accurate it actually is.
 *
 * Reporting only the in-distribution figure would be true and misleading at
 * once, so all three protocols are shown together — including the one where
 * the system is weakest. A user who is told 96% and then watches it get a
 * new merchant wrong has been misled; a user told that familiar merchants are
 * 96% and genuinely new text is 56% has been given something they can use.
 */
import React from 'react';
import {
  RefreshControl, ScrollView, View, useWindowDimensions,
} from 'react-native';

import { useTheme } from '../ThemeContext';
import { api } from '../api';
import {
  Card, Divider, Loading, Pill, Problem, Row, SectionTitle, ServerSvg, T,
  space,
} from '../components/ui';
import { layerColour } from '../theme';
import { useLoad } from '../useLoad';

const LAYER_NOTE: Record<string, string> = {
  correction: 'Something you corrected yourself. Checked before anything else.',
  rule: 'A banking phrase that leaves no room for doubt, such as OVERDRAFT.',
  memory: 'A merchant you have had a transaction with before.',
  model: 'Everything else — a sentence encoder with a confidence gate.',
};

export function HowItWorksScreen() {
  const { p } = useTheme();
  const { width } = useWindowDimensions();
  const state = useLoad(() => api.howItWorks(), []);
  const inner = width - space.lg * 4;
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
            Four layers, each answering only what it can account for and
            passing the rest down. A transaction the model is less than{' '}
            {Math.round(d.threshold * 100)}% sure of is not guessed at — it is
            held for you.
          </T>

          <SectionTitle note="Measured on your own transactions.">
            Which layer decided
          </SectionTitle>
          <Card padded={false}>
            {d.layers.map((layer, index) => (
              <View key={layer.layer}>
                {index ? <Divider /> : null}
                <Row>
                  <View style={{
                    flexDirection: 'row', justifyContent: 'space-between',
                    alignItems: 'baseline',
                  }}>
                    <T variant="body">
                      {layer.layer === 'memory' ? 'Merchant memory'
                        : layer.layer === 'correction' ? 'Your corrections'
                        : layer.layer === 'rule' ? 'Rules' : 'The model'}
                    </T>
                    <T variant="small" tone="muted">
                      {layer.count.toLocaleString()} · {layer.share.toFixed(1)}%
                    </T>
                  </View>
                  <View style={{
                    height: 8, borderRadius: 4, backgroundColor: p.surface2,
                    marginTop: 8, overflow: 'hidden',
                  }}>
                    <View style={{
                      width: `${layer.share}%`, height: '100%',
                      backgroundColor: layerColour(layer.layer, p),
                    }} />
                  </View>
                  <T variant="small" tone="faint" style={{ marginTop: 5 }}>
                    {LAYER_NOTE[layer.layer]}
                  </T>
                </Row>
              </View>
            ))}
          </Card>
        </>
      ) : null}
    </ScrollView>
  );
}
