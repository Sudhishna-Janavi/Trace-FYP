/**
 * Why one transaction carries the category it does — and the chance to
 * disagree.
 *
 * This is the screen the whole project exists for, so it is the one that
 * refuses to round anything off. The deciding layer is named. Whether its
 * reason is exact or reconstructed is stated in words, not implied by a
 * colour. The attribution bars are the model's own weights when the model
 * decided, and the one token a rule matched when a rule did. The time taken
 * is shown because an explanation that costs 350 ms is a different kind of
 * thing from one that costs under a millisecond, and hiding that would be a
 * small dishonesty about what the system is doing.
 */
import React, { useState } from 'react';
import { Modal, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useAuth } from '../AuthContext';
import { useTheme } from '../ThemeContext';
import { ApiError, Explanation, api } from '../api';
import {
  Button, Card, Divider, Loading, Money, Pill, Problem, Row, T, radius, space,
} from '../components/ui';
import { layerColour, layerSoft, type } from '../theme';
import { useLoad } from '../useLoad';

export function TransactionScreen({ route, navigation }: any) {
  const id: number = route.params.id;
  const { p } = useTheme();
  const { categories, refresh } = useAuth();
  const state = useLoad<Explanation>(() => api.explanation(id), [id]);
  const [picking, setPicking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  async function apply(category: string | null) {
    setProblem(null);
    setSaving(true);
    try {
      const result = category === null
        ? await api.confirm(id)
        : await api.correct(id, category);
      setSaved(result.corrected
        ? `Saved. ${result.also_updated} other ${result.also_updated === 1 ? 'transaction' : 'transactions'} at this merchant updated.`
        : 'Confirmed.');
      refresh();
      state.reload();
    } catch (error) {
      setProblem(error instanceof ApiError ? error.message : 'That did not save.');
    } finally {
      setSaving(false);
      setPicking(false);
    }
  }

  const d = state.data;

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['bottom']}>
      <ScrollView contentContainerStyle={{ padding: space.lg, paddingBottom: 40 }}>
        {state.loading ? <Loading note="Working out why…" /> : null}
        {state.error ? <Problem message={state.error} onRetry={state.reload} /> : null}

        {d ? (
          <>
            <T variant="title">{d.category}</T>
            <T variant="mono" tone="muted" style={{ marginTop: 8 }}>
              {d.description}
            </T>

            <Card style={{ marginTop: space.lg, padding: 0 }}>
              <Fact label="Amount"><Money value={d.amount} variant="body" /></Fact>
              <Divider />
              <Fact label="Date"><T variant="body">{d.date}</T></Fact>
              <Divider />
              <Fact label="Account"><T variant="body">{d.bank}</T></Fact>
              <Divider />
              <Fact label="Confidence">
                <T variant="body"
                   style={{ color: d.confidence >= 0.6 ? p.ink : p.review }}>
                  {Math.round(d.confidence * 100)}%
                </T>
              </Fact>
            </Card>

            <T variant="label" tone="faint"
               style={{ marginTop: space.xl, letterSpacing: 0.8 }}>
              WHY THIS CATEGORY
            </T>

            <View style={{ flexDirection: 'row', gap: 7, marginTop: space.md }}>
              <View style={{
                backgroundColor: layerSoft(d.layer, p),
                borderRadius: radius.pill, paddingHorizontal: 10, paddingVertical: 5,
              }}>
                <T variant="label" style={{ color: layerColour(d.layer, p) }}>
                  {d.exact ? '✓ Exact' : '⚠ Approximate'}
                </T>
              </View>
              <Pill label={d.layer_label} />
            </View>

            <View style={{
              marginTop: space.md,
              backgroundColor: p.surface2,
              borderRadius: radius.card,
              padding: space.lg,
            }}>
              <T variant="body">{d.headline}</T>
            </View>

            {d.tokens.length ? (
              <View style={{ marginTop: space.lg, gap: 10 }}>
                {d.tokens.map(([token, weight]) => (
                  <TokenBar key={token} token={token} weight={weight}
                            peak={Math.max(...d.tokens.map(t => Math.abs(t[1])), 0.0001)} />
                ))}
              </View>
            ) : null}

            <T variant="small" tone="muted" style={{ marginTop: space.lg }}>
              {d.note}
            </T>
            <T variant="small" tone="faint" style={{ marginTop: 6 }}>
              Computed in {d.took_ms} ms.
            </T>

            <T variant="label" tone="faint"
               style={{ marginTop: space.xxl, letterSpacing: 0.8 }}>
              IS THIS RIGHT?
            </T>

            {saved ? (
              <View style={{
                marginTop: space.md, padding: space.lg,
                backgroundColor: p.accentSoft, borderRadius: radius.card,
              }}>
                <T variant="small" style={{ color: p.accent }}>{saved}</T>
              </View>
            ) : null}
            {problem ? (
              <View style={{ marginTop: space.md }}>
                <Problem message={problem} />
              </View>
            ) : null}

            <Button
              label={`Yes, ${d.category} is correct`}
              kind="ghost"
              busy={saving}
              onPress={() => apply(null)}
              style={{ marginTop: space.md }}
            />
            <Button
              label="Change the category"
              kind="quiet"
              busy={saving}
              onPress={() => setPicking(true)}
              style={{ marginTop: space.sm }}
            />

            <T variant="small" tone="faint" style={{ marginTop: space.lg }}>
              A correction is remembered for this merchant and applied to your
              other transactions there. It is checked before any rule or model
              next time.
            </T>
          </>
        ) : null}
      </ScrollView>

      <CategoryPicker
        visible={picking}
        categories={categories}
        current={d?.category}
        onPick={apply}
        onClose={() => setPicking(false)}
      />
    </SafeAreaView>
  );
}

/* ------------------------------------------------------------------ bits -- */

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Row>
      <View style={{
        flexDirection: 'row', justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <T variant="small" tone="muted">{label}</T>
        {children}
      </View>
    </Row>
  );
}

/**
 * One term's contribution.
 *
 * The bar is proportional to the largest contribution in the explanation, not
 * to a fixed scale, because what a reader wants from it is the ordering and
 * the relative size — and the signed number beside it carries the magnitude
 * exactly for anyone who wants that instead.
 */
function TokenBar({ token, weight, peak }: {
  token: string; weight: number; peak: number;
}) {
  const { p } = useTheme();
  const positive = weight >= 0;
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: space.md }}>
      <T variant="monoSmall" tone="muted" numberOfLines={1}
         style={{ width: 92 }}>
        {token}
      </T>
      <View style={{
        flex: 1, height: 7, borderRadius: 3.5,
        backgroundColor: p.surface2, overflow: 'hidden',
      }}>
        <View style={{
          width: `${Math.max(2, (Math.abs(weight) / peak) * 100)}%`,
          height: '100%',
          backgroundColor: positive ? p.accent : p.spend,
        }} />
      </View>
      <T variant="monoSmall" tone="faint" style={{ width: 46, textAlign: 'right' }}>
        {positive ? '+' : '−'}{Math.abs(weight).toFixed(2)}
      </T>
    </View>
  );
}

function CategoryPicker({ visible, categories, current, onPick, onClose }: {
  visible: boolean;
  categories: string[];
  current?: string;
  onPick: (category: string) => void;
  onClose: () => void;
}) {
  const { p } = useTheme();
  return (
    <Modal visible={visible} animationType="slide" transparent
           onRequestClose={onClose}>
      <Pressable
        onPress={onClose}
        style={{ flex: 1, backgroundColor: 'rgba(12,13,12,0.45)' }}
      />
      <View style={{
        backgroundColor: p.surface,
        borderTopLeftRadius: 18, borderTopRightRadius: 18,
        paddingBottom: 28, maxHeight: '72%',
      }}>
        <View style={{ alignItems: 'center', paddingVertical: 10 }}>
          <View style={{
            width: 38, height: 4, borderRadius: 2, backgroundColor: p.line,
          }} />
        </View>
        <View style={{ paddingHorizontal: space.lg, paddingBottom: space.md }}>
          <T variant="heading">Choose a category</T>
        </View>
        <View style={{ height: StyleSheet.hairlineWidth, backgroundColor: p.line }} />
        <ScrollView>
          {categories.map(category => (
            <Pressable
              key={category}
              onPress={() => onPick(category)}
              style={({ pressed }) => ({
                paddingVertical: 14, paddingHorizontal: space.lg,
                minHeight: 46, justifyContent: 'center',
                backgroundColor: pressed ? p.surface2 : 'transparent',
              })}>
              <View style={{
                flexDirection: 'row', justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                <T variant="body"
                   style={category === current
                     ? { color: p.accent, fontFamily: type.heading.fontFamily }
                     : undefined}>
                  {category}
                </T>
                {category === current ? (
                  <T variant="body" style={{ color: p.accent }}>{'✓'}</T>
                ) : null}
              </View>
            </Pressable>
          ))}
        </ScrollView>
      </View>
    </Modal>
  );
}
