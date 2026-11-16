/**
 * The ledger: every transaction, from every account, in one list.
 *
 * On the web this is a table. A table does not survive a 390-point screen, so
 * here each row is a card with the description on top, the amount against it,
 * and the date, the account and the deciding layer beneath. The information
 * is identical and the hierarchy is the same; only the arrangement changed.
 *
 * The search runs on the server rather than filtering a page already in hand,
 * because the list is paged and filtering locally would quietly search only
 * the hundred rows that happen to be loaded.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator, FlatList, Pressable, ScrollView, StyleSheet, View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useTheme } from '../ThemeContext';
import { ApiError, Tx, api } from '../api';
import {
  Divider, Field, Loading, Problem, T, radius, space,
} from '../components/ui';
import { LedgerRow } from './Dashboard';

type State = '' | 'review' | 'confirmed';

export function TransactionsScreen({ navigation }: any) {
  const { p } = useTheme();
  const [query, setQuery] = useState('');
  const [typed, setTyped] = useState('');
  const [state, setState] = useState<State>('');
  const [category, setCategory] = useState('');

  const [rows, setRows] = useState<Tx[]>([]);
  const [page, setPage] = useState(1);
  const [pages, setPages] = useState(1);
  const [matching, setMatching] = useState(0);
  const [total, setTotal] = useState(0);
  const [categories, setCategories] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Searching on every keystroke would send a request per letter. A short
  // pause after typing stops is enough to feel immediate and asks once.
  useEffect(() => {
    const timer = setTimeout(() => setQuery(typed), 320);
    return () => clearTimeout(timer);
  }, [typed]);

  const load = useCallback(async (wanted: number, append: boolean) => {
    if (append) setMore(true); else setLoading(true);
    setError(null);
    try {
      const result = await api.transactions({
        q: query, category, state, page: wanted,
      });
      setRows(current => (append ? [...current, ...result.rows] : result.rows));
      setPage(result.page);
      setPages(result.pages);
      setMatching(result.matching);
      setTotal(result.total);
      setCategories(result.all_categories);
    } catch (problem) {
      setError(problem instanceof ApiError ? problem.message : 'Could not load the ledger.');
    } finally {
      setLoading(false);
      setMore(false);
    }
  }, [query, category, state]);

  useEffect(() => { load(1, false); }, [load]);

  const filtered = useMemo(
    () => Boolean(query || category || state), [query, category, state]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: p.bg }} edges={['top']}>
      <View style={{ paddingHorizontal: space.lg, paddingTop: space.sm }}>
        <T variant="display">Transactions</T>
        <T variant="small" tone="muted" style={{ marginTop: 4 }}>
          {filtered
            ? `${matching.toLocaleString()} of ${total.toLocaleString()} transactions`
            : `${total.toLocaleString()} transactions`}
        </T>

        <Field
          label=""
          value={typed}
          onChangeText={setTyped}
          placeholder="Search descriptions…"
          autoCapitalize="none"
          autoCorrect={false}
          returnKeyType="search"
          clearButtonMode="while-editing"
        />

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={{ gap: 7, paddingVertical: space.md }}>
          <Chip label="All" active={state === ''} onPress={() => setState('')} />
          <Chip label="Needs review" active={state === 'review'}
                onPress={() => setState('review')} />
          <Chip label="Confirmed" active={state === 'confirmed'}
                onPress={() => setState('confirmed')} />
          {category ? (
            <Chip label={`${category} ×`} active
                  onPress={() => setCategory('')} />
          ) : null}
        </ScrollView>

        {!category && categories.length ? (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ gap: 7, paddingBottom: space.md }}>
            {categories.map(c => (
              <Chip key={c} label={c} active={false} quiet
                    onPress={() => setCategory(c)} />
            ))}
          </ScrollView>
        ) : null}
      </View>

      {error ? (
        <View style={{ padding: space.lg }}>
          <Problem message={error} onRetry={() => load(1, false)} />
        </View>
      ) : null}

      {loading ? <Loading /> : (
        <FlatList
          data={rows}
          keyExtractor={row => String(row.id)}
          ItemSeparatorComponent={Divider}
          contentContainerStyle={{
            backgroundColor: p.surface,
            marginHorizontal: space.lg,
            borderRadius: radius.large,
            borderWidth: StyleSheet.hairlineWidth,
            borderColor: p.line,
            overflow: 'hidden',
            marginBottom: space.lg,
          }}
          renderItem={({ item }) => (
            <LedgerRow
              row={item}
              onPress={() => navigation.navigate('Transaction', { id: item.id })}
            />
          )}
          ListEmptyComponent={
            <View style={{ padding: space.xl }}>
              <T variant="body" tone="muted">
                Nothing matches those filters.
              </T>
            </View>
          }
          onEndReachedThreshold={0.4}
          onEndReached={() => {
            if (!more && page < pages) load(page + 1, true);
          }}
          ListFooterComponent={
            more ? (
              <View style={{ padding: space.lg }}>
                <ActivityIndicator color={p.muted} />
              </View>
            ) : page < pages ? (
              <View style={{ padding: space.lg }}>
                <T variant="small" tone="faint">
                  Showing {rows.length.toLocaleString()} of {matching.toLocaleString()}
                </T>
              </View>
            ) : undefined
          }
        />
      )}
    </SafeAreaView>
  );
}

function Chip({ label, active, quiet, onPress }: {
  label: string; active: boolean; quiet?: boolean; onPress: () => void;
}) {
  const { p } = useTheme();
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => ({
        backgroundColor: active ? p.accent : quiet ? 'transparent' : p.surface,
        borderColor: active ? p.accent : p.line,
        borderWidth: StyleSheet.hairlineWidth,
        borderRadius: radius.pill,
        paddingHorizontal: 13,
        height: 34,
        justifyContent: 'center',
        opacity: pressed ? 0.7 : 1,
      })}>
      <T variant="label" style={{ color: active ? p.onAccent : p.muted }}>
        {label}
      </T>
    </Pressable>
  );
}
