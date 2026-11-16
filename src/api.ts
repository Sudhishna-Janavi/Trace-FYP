/**
 * The client for Trace's JSON API.
 *
 * One place that knows how to talk to the server: where it is, how the token
 * is attached, how long to wait, and what an error means. Screens call named
 * functions and never see a URL.
 *
 * The token is the same signed value the web application keeps in a cookie.
 * On the device it is held in the system keychain through expo-secure-store,
 * which is encrypted at rest and not readable by other applications. On the
 * web target — used only for development — there is no keychain, so it falls
 * back to localStorage, and that build is never given to anyone.
 */
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

import { API_BASE, REQUEST_TIMEOUT_MS } from './config';

const TOKEN_KEY = 'trace.session';

/* ------------------------------------------------------------- the token -- */

export async function getToken(): Promise<string | null> {
  if (Platform.OS === 'web') {
    try {
      return globalThis.localStorage?.getItem(TOKEN_KEY) ?? null;
    } catch {
      return null;
    }
  }
  return SecureStore.getItemAsync(TOKEN_KEY);
}

export async function setToken(token: string): Promise<void> {
  if (Platform.OS === 'web') {
    try {
      globalThis.localStorage?.setItem(TOKEN_KEY, token);
    } catch {
      /* a private window refuses; the session simply does not persist */
    }
    return;
  }
  await SecureStore.setItemAsync(TOKEN_KEY, token);
}

export async function clearToken(): Promise<void> {
  if (Platform.OS === 'web') {
    try {
      globalThis.localStorage?.removeItem(TOKEN_KEY);
    } catch {
      /* nothing to clear */
    }
    return;
  }
  await SecureStore.deleteItemAsync(TOKEN_KEY);
}

/* ---------------------------------------------------------------- errors -- */

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

/** True when the token is gone or rejected, so the caller should sign out. */
export const isUnauthorised = (error: unknown) =>
  error instanceof ApiError && error.status === 401;

/* --------------------------------------------------------------- request -- */

interface Options {
  method?: 'GET' | 'POST';
  body?: unknown;
  form?: FormData;
  auth?: boolean;
}

async function request<T>(path: string, options: Options = {}): Promise<T> {
  const { method = 'GET', body, form, auth = true } = options;

  const headers: Record<string, string> = { Accept: 'application/json' };
  if (auth) {
    const token = await getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  // FormData sets its own content type, including the multipart boundary;
  // setting it by hand produces a body the server cannot parse.
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);
    if ((error as Error)?.name === 'AbortError') {
      throw new ApiError(0, 'The server took too long to answer. It may be '
        + 'waking up — try once more.');
    }
    throw new ApiError(0, 'Could not reach Trace. Check the connection.');
  }
  clearTimeout(timer);

  if (!response.ok) {
    // FastAPI puts the message in `detail`; anything else is unexpected and
    // should not be shown to a user as-is.
    let message = 'Something went wrong.';
    try {
      const payload = await response.json();
      if (typeof payload?.detail === 'string') message = payload.detail;
      else if (typeof payload?.error === 'string') message = payload.error;
    } catch {
      /* a non-JSON body, such as a proxy error page */
    }
    // A 401 means two different things. On a request that carried a token it
    // means the token is no longer good, and saying so is helpful. On sign-in
    // it means the password was wrong, and "your session has expired" would
    // be both false and baffling — so the server's own wording stands.
    if (response.status === 401 && auth) message = 'Your session has expired.';
    throw new ApiError(response.status, message);
  }

  return (await response.json()) as T;
}

/* ----------------------------------------------------------------- types -- */

export interface User {
  id: number;
  display_name: string;
  email: string;
  theme: string;
}

export interface Session {
  token: string;
  user: User;
  has_data: boolean;
}

export interface Me {
  user: User;
  has_data: boolean;
  transactions: number;
  review_count: number;
  categories: string[];
  threshold: number;
}

export interface Tx {
  id: number;
  description: string;
  merchant: string;
  amount: number;
  date: string;
  account_id: number;
  bank: string;
  category: string;
  predicted_category: string;
  corrected_category: string | null;
  confidence: number;
  layer: string;
  layer_label: string;
  needs_review: boolean;
  reviewed: boolean;
}

export interface Totals {
  spending: number;
  income: number;
  net: number;
  transactions: number;
  needs_review: number;
  accounts: number;
  first: string | null;
  last: string | null;
}

export interface CategorySlice {
  category: string;
  amount: number;
  count: number;
  share: number;
}

export interface Recurring {
  merchant: string;
  category: string;
  description: string;
  occurrences: number;
  average: number;
  interval: number;
  annual: number;
}

export interface Dashboard {
  has_data: boolean;
  greeting: string;
  totals: Totals;
  months: { month: string; label: string; spending: number; income: number }[];
  chart: string;
  categories: CategorySlice[];
  recurring: Recurring[];
  recent: Tx[];
  review_count: number;
}

export interface TransactionPage {
  rows: Tx[];
  matching: number;
  total: number;
  page: number;
  pages: number;
  per_page: number;
  accounts: Account[];
  all_categories: string[];
  filters: { q: string; category: string; account: string; state: string };
}

export interface Account {
  id: number;
  bank: string;
  name: string;
  source_format: string;
  currency: string;
  external_id: string;
}

export interface Explanation {
  id: number;
  description: string;
  amount: number;
  date: string;
  bank: string;
  category: string;
  confidence: number;
  layer: string;
  layer_label: string;
  method: string;
  exact: boolean;
  headline: string;
  tokens: [string, number][];
  note: string;
  took_ms: number;
}

export interface ReviewResult {
  transaction_id: number;
  category: string;
  corrected: boolean;
  also_updated: number;
  remaining: number;
}

export interface ReviewQueue {
  rows: Tx[];
  reviewed: number;
  threshold: number;
  categories: string[];
  corrections: {
    merchant: string;
    category: string;
    times_applied: number;
    created_at: string;
  }[];
}

export interface IngestSummary {
  transactions: number;
  accounts: number;
  files: number;
  needs_review: number;
  by_layer: Record<string, number>;
  layers: { key: string; label: string; note: string; count: number }[];
  uploaded?: {
    files: number;
    banks: string[];
    banks_phrase: string;
    rejected: { name: string; why: string }[];
  };
}

export interface Insights {
  has_data: boolean;
  categories: CategorySlice[];
  wheel: string;
  segments: { category: string; amount: number; share: number; colour?: string }[];
  recurring: Recurring[];
  unusual: {
    id: number;
    description: string;
    category: string;
    date: string;
    amount: number;
    baseline: number;
    score: number;
  }[];
  change: {
    available: boolean;
    month: string | null;
    previous: string | null;
    months?: number;
    rows: {
      category: string;
      now: number;
      previous: number;
      average: number;
      vs_previous: number;
      vs_average: number;
    }[];
  };
  change_chart: string;
  change_chart_prev: string;
  signals: {
    months: number;
    lowest: number;
    highest: number;
    average: number;
    trend_per_month: number;
    overspent_months: number;
    overspent_list: string[];
  } | null;
  benchmark: {
    category: string;
    amount: number;
    share: number;
    benchmark: number;
    gap: number;
  }[];
  methodology: string[];
  is_demo: boolean;
}

export interface AccountsPayload {
  detail: {
    account: Account;
    rows: number;
    first: string | null;
    last: string | null;
    spend: number;
    share: number;
  }[];
  total: number;
}

export interface HowItWorks {
  card: Record<string, any>;
  performance: Record<string, Record<string, number>>;
  labels: Record<string, string>;
  means: Record<string, string>;
  chart: string;
  layers: { layer: string; count: number; share: number }[];
  total: number;
  flagged: number;
  observed_rate: number;
  explain_cost: { layer: string; method: string; exact: boolean; cost: string }[];
  threshold: number;
}

/* ------------------------------------------------------------- endpoints -- */

export const api = {
  signUp: (display_name: string, email: string, password: string) =>
    request<Session>('/auth/signup', {
      method: 'POST', auth: false, body: { display_name, email, password },
    }),

  signIn: (email: string, password: string) =>
    request<Session>('/auth/signin', {
      method: 'POST', auth: false, body: { email, password },
    }),

  me: () => request<Me>('/me'),

  setTheme: (theme: string) =>
    request<{ theme: string }>('/theme', { method: 'POST', body: { theme } }),

  loadDemo: () => request<IngestSummary>('/connect/demo', { method: 'POST' }),

  upload: (files: { uri: string; name: string; mimeType?: string }[]) => {
    const form = new FormData();
    for (const file of files) {
      // React Native's FormData takes this shape rather than a Blob; the
      // field name must match the server's `statements` parameter.
      form.append('statements', {
        uri: file.uri,
        name: file.name,
        type: file.mimeType || 'text/csv',
      } as any);
    }
    return request<IngestSummary>('/connect/upload', { method: 'POST', form });
  },

  dashboard: () => request<Dashboard>('/dashboard'),

  transactions: (params: {
    q?: string; category?: string; account?: string; state?: string; page?: number;
  } = {}) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') query.set(key, String(value));
    }
    const suffix = query.toString();
    return request<TransactionPage>(`/transactions${suffix ? `?${suffix}` : ''}`);
  },

  explanation: (id: number, method: 'shap' | 'lime' = 'shap') =>
    request<Explanation>(`/transactions/${id}/explanation?method=${method}`),

  confirm: (id: number) =>
    request<ReviewResult>(`/transactions/${id}/confirm`, { method: 'POST' }),

  correct: (id: number, category: string) =>
    request<ReviewResult>(`/transactions/${id}/category`, {
      method: 'POST', body: { category },
    }),

  review: () => request<ReviewQueue>('/review'),
  insights: () => request<Insights>('/insights'),
  accounts: () => request<AccountsPayload>('/accounts'),
  howItWorks: () => request<HowItWorks>('/how-it-works'),
};
