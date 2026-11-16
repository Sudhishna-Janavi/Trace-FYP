/**
 * Trace's design tokens, carried over unchanged from the web application.
 *
 * Every value here is copied from `backend/static/trace.css`. Nothing is
 * approximated and nothing is "close enough": the phone and the browser draw
 * the same product, so they use the same hexadecimal values. If a token
 * changes there it must change here, which is why each one is named after the
 * CSS custom property it mirrors.
 *
 * Colour is semantic, as it is on the web: the accent marks a decision the
 * system can account for exactly, amber marks one that needs a person's
 * attention, terracotta is money leaving.
 */

export type Scheme = 'light' | 'dark';

export interface Palette {
  bg: string;
  surface: string;
  surface2: string;
  ink: string;
  muted: string;
  faint: string;
  line: string;
  accent: string;
  accentSoft: string;
  onAccent: string;
  income: string;
  spend: string;
  review: string;
  reviewSoft: string;
  danger: string;
  dangerSoft: string;
}

/** `:root` in trace.css */
export const light: Palette = {
  bg: '#FBFBFA',
  surface: '#FFFFFF',
  surface2: '#F6F7F5',
  ink: '#1A1C1B',
  muted: '#6B7471',
  faint: '#9BA5A1',
  line: '#E6E8E6',
  accent: '#1A6B5C',
  accentSoft: '#E8F1EE',
  onAccent: '#FFFFFF',
  income: '#1F7A4D',
  spend: '#B04A33',
  review: '#9A6B14',
  reviewSoft: '#FBF3E0',
  danger: '#B04A33',
  dangerSoft: '#F8ECE8',
};

/** `:root[data-theme="dark"]` in trace.css */
export const dark: Palette = {
  bg: '#0C0D0C',
  surface: '#161817',
  surface2: '#1E211F',
  ink: '#ECEEED',
  muted: '#9BA5A1',
  faint: '#6B7471',
  line: '#262927',
  accent: '#4DB39B',
  accentSoft: '#17302A',
  onAccent: '#0C0D0C',
  income: '#4ADE80',
  spend: '#F08A6C',
  review: '#E3B341',
  reviewSoft: '#2A2312',
  danger: '#F08A6C',
  dangerSoft: '#2A1713',
};

export const palettes: Record<Scheme, Palette> = { light, dark };

/** `--r` and `--r-lg` */
export const radius = { card: 10, large: 14, pill: 999 } as const;

export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 22, xxl: 32 } as const;

/**
 * The web application sets its body text in IBM Plex Sans and its transaction
 * descriptions in IBM Plex Mono. A bank description is a fixed-width string
 * from a machine, and setting it in a proportional face makes two similar
 * descriptions hard to tell apart — which matters on a screen whose entire
 * purpose is explaining why two similar descriptions were categorised
 * differently.
 */
export const font = {
  regular: 'IBMPlexSans_400Regular',
  medium: 'IBMPlexSans_500Medium',
  semibold: 'IBMPlexSans_600SemiBold',
  bold: 'IBMPlexSans_700Bold',
  mono: 'IBMPlexMono_400Regular',
} as const;

export const type = {
  display: { fontFamily: font.semibold, fontSize: 30, letterSpacing: -0.6 },
  title: { fontFamily: font.semibold, fontSize: 21, letterSpacing: -0.4 },
  heading: { fontFamily: font.semibold, fontSize: 16, letterSpacing: -0.2 },
  body: { fontFamily: font.regular, fontSize: 15, lineHeight: 22 },
  small: { fontFamily: font.regular, fontSize: 13, lineHeight: 19 },
  label: { fontFamily: font.medium, fontSize: 12.5 },
  mono: { fontFamily: font.mono, fontSize: 13 },
  monoSmall: { fontFamily: font.mono, fontSize: 11.5 },
} as const;

/**
 * Which colour a cascade layer is drawn in.
 *
 * The three deterministic layers share the accent because the distinction
 * that matters to a reader is exact against approximate, not rule against
 * memory. The model gets amber: it is the only layer that can be wrong.
 */
export function layerColour(layer: string, p: Palette): string {
  return layer === 'model' ? p.review : p.accent;
}

export function layerSoft(layer: string, p: Palette): string {
  return layer === 'model' ? p.reviewSoft : p.accentSoft;
}

/** `−$1,234.56`, with a true minus sign, exactly as the `money` filter does. */
export function money(value: number, decimals: 0 | 2 = 2): string {
  const sign = value < 0 ? '−' : '';
  const body = Math.abs(value).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return `${sign}$${body}`;
}

/** `30 Sep 2026` from the API's ISO date. */
export function dateLabel(iso: string): string {
  const [y, m, d] = iso.split('-').map(Number);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${d} ${months[m - 1]} ${y}`;
}
